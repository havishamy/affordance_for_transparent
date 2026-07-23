from __future__ import annotations

import cv2
import numpy as np

from primitive_recovery.geometry import CameraIntrinsics, project_camera_to_pixel, rpy_to_rotation_matrix
from primitive_recovery.templates_jar import JarParams


def _sample_profile_points(params: JarParams, n_theta: int = 180) -> np.ndarray:
    total_height = params.body_height + params.shoulder_height + params.neck_height + params.lip_height
    y_bottom = 0.5 * total_height
    y_body_top = y_bottom - params.body_height
    y_shoulder_top = y_body_top - params.shoulder_height
    y_neck_top = y_shoulder_top - params.neck_height
    y_lip_top = y_neck_top - params.lip_height

    n_body = max(24, int(round(params.body_height / max(total_height, 1e-6) * 120)))
    n_shoulder = max(16, int(round(params.shoulder_height / max(total_height, 1e-6) * 80)))
    n_neck = max(16, int(round(params.neck_height / max(total_height, 1e-6) * 60)))
    n_lip = max(10, int(round(params.lip_height / max(total_height, 1e-6) * 32)))

    ys = []
    rs = []

    body_ys = np.linspace(y_bottom, y_body_top, n_body, endpoint=False, dtype=np.float32)
    body_rs = np.full_like(body_ys, params.body_radius)
    ys.append(body_ys)
    rs.append(body_rs)

    shoulder_ys = np.linspace(y_body_top, y_shoulder_top, n_shoulder, endpoint=False, dtype=np.float32)
    shoulder_t = np.linspace(0.0, 1.0, n_shoulder, endpoint=False, dtype=np.float32)
    shoulder_rs = params.body_radius + (params.neck_radius - params.body_radius) * shoulder_t
    ys.append(shoulder_ys)
    rs.append(shoulder_rs)

    neck_ys = np.linspace(y_shoulder_top, y_neck_top, n_neck, endpoint=False, dtype=np.float32)
    neck_rs = np.full_like(neck_ys, params.neck_radius)
    ys.append(neck_ys)
    rs.append(neck_rs)

    lip_ys = np.linspace(y_neck_top, y_lip_top, n_lip, endpoint=True, dtype=np.float32)
    lip_t = np.linspace(0.0, 1.0, n_lip, endpoint=True, dtype=np.float32)
    lip_rs = params.neck_radius + (params.lip_radius - params.neck_radius) * np.sin(lip_t * np.pi * 0.5)
    ys.append(lip_ys)
    rs.append(lip_rs)

    ys = np.concatenate(ys, axis=0)
    rs = np.concatenate(rs, axis=0)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=np.float32)

    points = []
    for y, r in zip(ys, rs):
        x = r * np.cos(thetas)
        z_local = r * np.sin(thetas)
        points.append(np.stack([x, np.full_like(x, y), z_local], axis=1))

    top_rim = np.stack(
        [
            params.lip_radius * np.cos(thetas),
            np.full_like(thetas, y_lip_top),
            params.lip_radius * np.sin(thetas),
        ],
        axis=1,
    )
    bottom_rim = np.stack(
        [
            params.body_radius * np.cos(thetas),
            np.full_like(thetas, y_bottom),
            params.body_radius * np.sin(thetas),
        ],
        axis=1,
    )
    points.extend([top_rim, bottom_rim])
    return np.concatenate(points, axis=0).astype(np.float32)


def _transform(points_local: np.ndarray, params: JarParams) -> np.ndarray:
    rot = rpy_to_rotation_matrix(params.roll, params.pitch, params.yaw)
    pts = (rot @ points_local.T).T
    pts += np.asarray([params.x, params.y, params.z], dtype=np.float32)
    return pts


def _project(points_xyz: np.ndarray, intrinsics: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    pix = []
    depth = []
    for p in points_xyz:
        uv = project_camera_to_pixel(p, intrinsics)
        if uv is None:
            continue
        pix.append(uv)
        depth.append(float(p[2]))
    if not pix:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(pix, dtype=np.float32), np.asarray(depth, dtype=np.float32)


def render_jar_mask(params: JarParams, image_shape: tuple[int, int], intrinsics: CameraIntrinsics) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    points_local = _sample_profile_points(params)
    points_world = _transform(points_local, params)
    pix, _ = _project(points_world, intrinsics)
    if len(pix) == 0:
        return mask

    pix_i = np.round(pix).astype(np.int32)
    pix_i[:, 0] = np.clip(pix_i[:, 0], 0, w - 1)
    pix_i[:, 1] = np.clip(pix_i[:, 1], 0, h - 1)
    hull = cv2.convexHull(pix_i.reshape(-1, 1, 2))
    cv2.fillConvexPoly(mask, hull[:, 0, :], 1)
    return mask


def render_jar_depth(params: JarParams, image_shape: tuple[int, int], intrinsics: CameraIntrinsics) -> np.ndarray:
    h, w = image_shape
    depth = np.zeros((h, w), dtype=np.float32)

    points_local = _sample_profile_points(params, n_theta=220)
    points_world = _transform(points_local, params)
    pix, z = _project(points_world, intrinsics)
    if len(pix) == 0:
        return depth

    pix_i = np.round(pix).astype(np.int32)
    valid = (
        (pix_i[:, 0] >= 0)
        & (pix_i[:, 0] < w)
        & (pix_i[:, 1] >= 0)
        & (pix_i[:, 1] < h)
    )
    pix_i = pix_i[valid]
    z = z[valid]
    for (u, v), zz in zip(pix_i, z):
        current = depth[v, u]
        if current == 0 or zz < current:
            depth[v, u] = zz

    mask = render_jar_mask(params, image_shape, intrinsics)
    valid_u8 = (depth > 0).astype(np.uint8)
    if valid_u8.sum() > 0:
        _, labels = cv2.distanceTransformWithLabels(1 - valid_u8, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        values = depth[valid_u8 > 0]
        dense = np.zeros_like(depth)
        for y in range(h):
            for x in range(w):
                if mask[y, x] == 0:
                    continue
                if depth[y, x] > 0:
                    dense[y, x] = depth[y, x]
                else:
                    idx = labels[y, x] - 1
                    if 0 <= idx < len(values):
                        dense[y, x] = values[idx]
        depth = dense

    depth *= mask.astype(np.float32)
    return depth
