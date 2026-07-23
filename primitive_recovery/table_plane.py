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


def _build_bottom_table_candidate_mask(
    object_mask: np.ndarray,
    side_pad: int = 28,
    down_pad: int = 70,
    bottom_band_ratio: float = 0.16,
    lateral_band_ratio: float = 0.22,
) -> np.ndarray:
    h, w = object_mask.shape
    ys, xs = np.where(object_mask > 0)
    if len(xs) == 0:
        return np.zeros_like(object_mask, dtype=np.uint8)

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    obj_h = max(1, y2 - y1)
    obj_w = max(1, x2 - x1)

    bottom_band_h = max(10, int(round(obj_h * bottom_band_ratio)))
    lateral_band_w = max(8, int(round(obj_w * lateral_band_ratio)))

    candidate = np.zeros_like(object_mask, dtype=np.uint8)

    # Region directly below the bottom footprint.
    xb1 = max(0, x1 - side_pad)
    xb2 = min(w, x2 + side_pad)
    yb1 = min(h, y2)
    yb2 = min(h, y2 + down_pad)
    if yb2 > yb1 and xb2 > xb1:
        candidate[yb1:yb2, xb1:xb2] = 1

    # Small lateral support bands near the lower sides of the object.
    yl1 = max(0, y2 - bottom_band_h)
    yl2 = min(h, y2 + max(12, down_pad // 2))

    xl1 = max(0, x1 - side_pad - lateral_band_w)
    xl2 = max(0, x1 + side_pad)
    if yl2 > yl1 and xl2 > xl1:
        candidate[yl1:yl2, xl1:xl2] = 1

    xr1 = min(w, x2 - side_pad)
    xr2 = min(w, x2 + side_pad + lateral_band_w)
    if yl2 > yl1 and xr2 > xr1:
        candidate[yl1:yl2, xr1:xr2] = 1

    candidate[object_mask > 0] = 0
    return candidate


def _oriented_plane_normal(normal: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    normal = normal.astype(np.float32)
    if points_xyz.shape[0] == 0:
        return normal
    centroid = points_xyz.mean(axis=0)
    if float(np.dot(normal, centroid)) > 0:
        normal = -normal
    return normal


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    out = np.zeros_like(mask, dtype=np.uint8)
    out[labels == idx] = 1
    return out


def estimate_table_plane_from_depth(
    object_mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_band_mm: float = 25.0,
    ransac_iters: int = 250,
    distance_thresh_mm: float = 4.0,
    min_points: int = 220,
    max_rms_mm: float = 3.5,
    min_component_ratio: float = 0.55,
    max_abs_normal_z: float = 0.75,
) -> TablePlaneEstimate | None:
    candidate_mask = _build_bottom_table_candidate_mask(object_mask)
    candidate_valid = (candidate_mask > 0) & (depth > 0)
    points_xyz, pixels_yx = _depth_to_points(depth, candidate_valid, intrinsics)
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

    inlier_mask = np.zeros_like(object_mask, dtype=np.uint8)
    inlier_pixels = pixels_yx[best_inliers]
    inlier_mask[inlier_pixels[:, 0], inlier_pixels[:, 1]] = 1
    largest_inlier_mask = _largest_component(inlier_mask)
    component_ratio = float(largest_inlier_mask.sum()) / max(float(inlier_mask.sum()), 1.0)
    if component_ratio < float(min_component_ratio):
        return None

    refined_points, refined_pixels = _depth_to_points(depth, largest_inlier_mask > 0, intrinsics)
    if refined_points.shape[0] < min_points:
        return None

    refined_normal, refined_offset = fit_plane_svd(refined_points)
    refined_normal = _oriented_plane_normal(refined_normal, refined_points)
    refined_offset = -float(np.dot(refined_normal, refined_points.mean(axis=0)))
    if abs(float(refined_normal[2])) > float(max_abs_normal_z):
        return None
    distances = point_plane_distance(refined_points, refined_normal, refined_offset)
    rms_error = float(np.sqrt(np.mean(distances**2))) if distances.size > 0 else 1e6
    if rms_error > float(max_rms_mm):
        return None

    return TablePlaneEstimate(
        normal=refined_normal.astype(np.float32),
        offset=float(refined_offset),
        inlier_mask=largest_inlier_mask.astype(np.uint8),
        ring_mask=candidate_mask.astype(np.uint8),
        point_count=int(refined_points.shape[0]),
        rms_error=rms_error,
        source="bottom_region_plane_ransac",
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
