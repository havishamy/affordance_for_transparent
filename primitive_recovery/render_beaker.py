from __future__ import annotations

import cv2
import numpy as np

from primitive_recovery.geometry import CameraIntrinsics, project_camera_to_pixel, rpy_to_rotation_matrix
from primitive_recovery.templates import BeakerParams


def _sample_beaker_points(
    params: BeakerParams,
    n_theta: int = 120,
    n_height: int = 60,
) -> np.ndarray:
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=np.float32)
    ys = np.linspace(-0.5 * params.height, 0.5 * params.height, n_height, dtype=np.float32)

    side_points = []
    for y in ys:
        x = params.radius * np.cos(thetas)
        z_local = params.radius * np.sin(thetas)
        pts = np.stack([x, np.full_like(x, y), z_local], axis=1)
        side_points.append(pts)
    side_points = np.concatenate(side_points, axis=0)

    # top and bottom rim circles
    x_circle = params.radius * np.cos(thetas)
    z_circle = params.radius * np.sin(thetas)
    top = np.stack([x_circle, np.full_like(x_circle, -0.5 * params.height), z_circle], axis=1)
    bottom = np.stack([x_circle, np.full_like(x_circle, 0.5 * params.height), z_circle], axis=1)

    pts = np.concatenate([side_points, top, bottom], axis=0).astype(np.float32)
    return pts


def _transform_points(points_local: np.ndarray, params: BeakerParams) -> np.ndarray:
    rot = rpy_to_rotation_matrix(params.roll, params.pitch, params.yaw)
    translated = (rot @ points_local.T).T
    translated += np.asarray([params.x, params.y, params.z], dtype=np.float32)
    return translated


def _project_points(points_xyz: np.ndarray, intrinsics: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    pixels = []
    depths = []
    for p in points_xyz:
        uv = project_camera_to_pixel(p, intrinsics)
        if uv is None:
            continue
        pixels.append(uv)
        depths.append(float(p[2]))
    if not pixels:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(pixels, dtype=np.float32), np.asarray(depths, dtype=np.float32)


def render_beaker_mask(
    params: BeakerParams,
    image_shape: tuple[int, int],
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    pts_local = _sample_beaker_points(params)
    pts_world = _transform_points(pts_local, params)
    pix, _ = _project_points(pts_world, intrinsics)
    if len(pix) == 0:
        return mask

    pix_i = np.round(pix).astype(np.int32)
    pix_i[:, 0] = np.clip(pix_i[:, 0], 0, width - 1)
    pix_i[:, 1] = np.clip(pix_i[:, 1], 0, height - 1)

    hull = cv2.convexHull(pix_i.reshape(-1, 1, 2))
    cv2.fillConvexPoly(mask, hull[:, 0, :], 1)
    return mask


def render_beaker_depth(
    params: BeakerParams,
    image_shape: tuple[int, int],
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    height, width = image_shape
    depth = np.zeros((height, width), dtype=np.float32)

    pts_local = _sample_beaker_points(params, n_theta=160, n_height=100)
    pts_world = _transform_points(pts_local, params)
    pix, depths = _project_points(pts_world, intrinsics)
    if len(pix) == 0:
        return depth

    pix_i = np.round(pix).astype(np.int32)
    valid = (
        (pix_i[:, 0] >= 0)
        & (pix_i[:, 0] < width)
        & (pix_i[:, 1] >= 0)
        & (pix_i[:, 1] < height)
    )
    pix_i = pix_i[valid]
    depths = depths[valid]

    # z-buffer like fill: keep closest surface point per pixel
    depth[:] = 0.0
    for (u, v), z in zip(pix_i, depths):
        current = depth[v, u]
        if current == 0 or z < current:
            depth[v, u] = z

    # densify sparse depth samples inside the rendered mask
    mask = render_beaker_mask(params, image_shape, intrinsics)
    sparse = depth.copy()
    valid_u8 = (sparse > 0).astype(np.uint8)
    if valid_u8.sum() > 0:
        nearest = cv2.distanceTransformWithLabels(1 - valid_u8, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        _, labels = nearest
        coords = np.argwhere(valid_u8 > 0)
        values = sparse[valid_u8 > 0]
        dense = np.zeros_like(sparse)
        for y in range(height):
            for x in range(width):
                if mask[y, x] == 0:
                    continue
                if sparse[y, x] > 0:
                    dense[y, x] = sparse[y, x]
                else:
                    idx = labels[y, x] - 1
                    if 0 <= idx < len(values):
                        dense[y, x] = values[idx]
        depth = dense

    depth *= mask.astype(np.float32)
    return depth
