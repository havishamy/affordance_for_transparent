from __future__ import annotations

import cv2
import numpy as np

from primitive_recovery.geometry import CameraIntrinsics, project_camera_to_pixel, rpy_to_rotation_matrix
from primitive_recovery.templates_jar import JarParams


def _sample_cylinder_points(radius: float, height: float, n_theta: int, n_height: int) -> np.ndarray:
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=np.float32)
    ys = np.linspace(-0.5 * height, 0.5 * height, n_height, dtype=np.float32)
    side_points = []
    for y in ys:
        x = radius * np.cos(thetas)
        z_local = radius * np.sin(thetas)
        pts = np.stack([x, np.full_like(x, y), z_local], axis=1)
        side_points.append(pts)
    side_points = np.concatenate(side_points, axis=0)
    return side_points.astype(np.float32)


def _sample_disk(radius: float, y: float, n_theta: int = 180) -> np.ndarray:
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=np.float32)
    x = radius * np.cos(thetas)
    z = radius * np.sin(thetas)
    return np.stack([x, np.full_like(x, y), z], axis=1).astype(np.float32)


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

    body_local = _sample_cylinder_points(params.body_radius, params.body_height, n_theta=160, n_height=80)
    lid_local = _sample_cylinder_points(params.lid_radius, params.lid_height, n_theta=120, n_height=20)

    body_world = _transform(body_local, params)
    lid_world = _transform(lid_local + np.asarray([0.0, -0.5 * params.body_height - 0.5 * params.lid_height, 0.0], dtype=np.float32), params)

    body_pix, _ = _project(body_world, intrinsics)
    lid_pix, _ = _project(lid_world, intrinsics)

    all_pix = []
    if len(body_pix) > 0:
        all_pix.append(body_pix)
    if len(lid_pix) > 0:
        all_pix.append(lid_pix)
    if not all_pix:
        return mask

    pix = np.concatenate(all_pix, axis=0)
    pix_i = np.round(pix).astype(np.int32)
    pix_i[:, 0] = np.clip(pix_i[:, 0], 0, w - 1)
    pix_i[:, 1] = np.clip(pix_i[:, 1], 0, h - 1)

    hull = cv2.convexHull(pix_i.reshape(-1, 1, 2))
    cv2.fillConvexPoly(mask, hull[:, 0, :], 1)
    return mask


def render_jar_depth(params: JarParams, image_shape: tuple[int, int], intrinsics: CameraIntrinsics) -> np.ndarray:
    h, w = image_shape
    depth = np.zeros((h, w), dtype=np.float32)

    body_local = _sample_cylinder_points(params.body_radius, params.body_height, n_theta=180, n_height=120)
    lid_local = _sample_cylinder_points(params.lid_radius, params.lid_height, n_theta=120, n_height=30)

    body_world = _transform(body_local, params)
    lid_world = _transform(lid_local + np.asarray([0.0, -0.5 * params.body_height - 0.5 * params.lid_height, 0.0], dtype=np.float32), params)

    for pts_world in [body_world, lid_world]:
        pix, z = _project(pts_world, intrinsics)
        if len(pix) == 0:
            continue
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
