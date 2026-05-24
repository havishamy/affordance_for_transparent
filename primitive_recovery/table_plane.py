from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from primitive_recovery.geometry import (
    CameraIntrinsics,
    backproject_pixel_to_camera,
    fit_plane_svd,
    point_plane_distance,
)


@dataclass
class TablePlaneEstimate:
    normal: np.ndarray
    offset: float
    inlier_mask: np.ndarray
    ring_mask: np.ndarray
    point_count: int
    rms_error: float
    source: str


def _depth_to_points(depth: np.ndarray, mask: np.ndarray, intrinsics: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)
    z = depth[ys, xs].astype(np.float32)
    valid = z > 0
    ys = ys[valid]
    xs = xs[valid]
    z = z[valid]
    if z.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)
    x = (xs.astype(np.float32) - intrinsics.cx) * z / max(intrinsics.fx, 1e-6)
    y = (ys.astype(np.float32) - intrinsics.cy) * z / max(intrinsics.fy, 1e-6)
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    pixels = np.stack([ys, xs], axis=1).astype(np.int32)
    return points, pixels


def _build_ring_mask(mask: np.ndarray, inner_pad: int, outer_pad: int) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outer_pad + 1, 2 * outer_pad + 1))
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inner_pad + 1, 2 * inner_pad + 1))
    outer = cv2.dilate(mask_u8, outer_kernel)
    inner = cv2.dilate(mask_u8, inner_kernel)
    ring = ((outer > 0) & (inner == 0)).astype(np.uint8)
    return ring


def _oriented_plane_normal(normal: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    normal = normal.astype(np.float32)
    if points_xyz.shape[0] == 0:
        return normal
    centroid = points_xyz.mean(axis=0)
    if float(np.dot(normal, centroid)) > 0:
        normal = -normal
    return normal


def estimate_table_plane_from_depth(
    object_mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    inner_pad: int = 10,
    outer_pad: int = 80,
    depth_band_mm: float = 35.0,
    ransac_iters: int = 250,
    distance_thresh_mm: float = 6.0,
    min_points: int = 500,
) -> TablePlaneEstimate | None:
    ring_mask = _build_ring_mask(object_mask, inner_pad=inner_pad, outer_pad=outer_pad)
    ring_valid = (ring_mask > 0) & (depth > 0)
    points_xyz, pixels_yx = _depth_to_points(depth, ring_valid, intrinsics)
    if points_xyz.shape[0] < min_points:
        return None

    z_values = points_xyz[:, 2]
    z_ref = float(np.median(z_values))
    keep = np.abs(z_values - z_ref) <= float(depth_band_mm)
    points_xyz = points_xyz[keep]
    pixels_yx = pixels_yx[keep]
    if points_xyz.shape[0] < min_points:
        return None

    rng = np.random.default_rng(42)
    best_inliers: np.ndarray | None = None
    best_normal: np.ndarray | None = None
    best_offset: float | None = None

    n_points = points_xyz.shape[0]
    for _ in range(max(ransac_iters, 1)):
        sample_idx = rng.choice(n_points, size=3, replace=False)
        sample = points_xyz[sample_idx]
        try:
            normal, offset = fit_plane_svd(sample)
        except np.linalg.LinAlgError:
            continue
        distances = point_plane_distance(points_xyz, normal, offset)
        inliers = distances <= float(distance_thresh_mm)
        if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers
            best_normal = normal
            best_offset = offset

    if best_inliers is None or best_normal is None or best_offset is None:
        return None
    if int(best_inliers.sum()) < min_points:
        return None

    refined_points = points_xyz[best_inliers]
    refined_normal, refined_offset = fit_plane_svd(refined_points)
    refined_normal = _oriented_plane_normal(refined_normal, refined_points)
    refined_offset = -float(np.dot(refined_normal, refined_points.mean(axis=0)))
    distances = point_plane_distance(refined_points, refined_normal, refined_offset)

    inlier_mask = np.zeros_like(object_mask, dtype=np.uint8)
    inlier_pixels = pixels_yx[best_inliers]
    inlier_mask[inlier_pixels[:, 0], inlier_pixels[:, 1]] = 1

    return TablePlaneEstimate(
        normal=refined_normal.astype(np.float32),
        offset=float(refined_offset),
        inlier_mask=inlier_mask,
        ring_mask=ring_mask.astype(np.uint8),
        point_count=int(refined_points.shape[0]),
        rms_error=float(np.sqrt(np.mean(distances**2))) if distances.size > 0 else 0.0,
        source="depth_plane_ransac",
    )


def estimate_cylinder_contact_point(
    object_mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray | None:
    ys, xs = np.where(object_mask > 0)
    if len(xs) == 0:
        return None
    y_bottom = int(ys.max())
    band_h = max(3, int(round(object_mask.shape[0] * 0.04)))
    y1 = max(0, y_bottom - band_h + 1)
    band = (object_mask[y1 : y_bottom + 1] > 0) & (depth[y1 : y_bottom + 1] > 0)
    band_ys, band_xs = np.where(band)
    if len(band_xs) == 0:
        return None
    band_ys = band_ys + y1
    z = float(np.median(depth[band_ys, band_xs]))
    u = float(np.median(band_xs))
    v = float(np.median(band_ys))
    return backproject_pixel_to_camera(u, v, z, intrinsics)
