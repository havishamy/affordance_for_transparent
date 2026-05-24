from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def backproject_pixel_to_camera(u: float, v: float, z: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return np.asarray([x, y, z], dtype=np.float32)


def project_camera_to_pixel(point_xyz: np.ndarray, intrinsics: CameraIntrinsics) -> tuple[float, float] | None:
    x, y, z = point_xyz
    if z <= 1e-6:
        return None
    u = intrinsics.fx * x / z + intrinsics.cx
    v = intrinsics.fy * y / z + intrinsics.cy
    return float(u), float(v)


def pixel_to_camera_ray(u: float, v: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    ray = np.asarray(
        [
            (u - intrinsics.cx) / max(intrinsics.fx, 1e-6),
            (v - intrinsics.cy) / max(intrinsics.fy, 1e-6),
            1.0,
        ],
        dtype=np.float32,
    )
    return ray


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def rotation_matrix_to_rpy(rot: np.ndarray) -> tuple[float, float, float]:
    value = float(np.clip(-rot[2, 0], -1.0, 1.0))
    pitch = math.asin(value)
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) > 1e-6:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-rot[0, 1]), float(rot[1, 1]))
    return float(roll), float(pitch), float(yaw)


def normalize_vector(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def rotation_matrix_from_y_axis(y_axis: np.ndarray, x_hint: np.ndarray | None = None) -> np.ndarray:
    y_axis = normalize_vector(y_axis)
    if x_hint is None:
        x_hint = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    x_axis = x_hint - np.dot(x_hint, y_axis) * y_axis
    if float(np.linalg.norm(x_axis)) < 1e-5:
        x_hint = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        x_axis = x_hint - np.dot(x_hint, y_axis) * y_axis
    x_axis = normalize_vector(x_axis)
    z_axis = normalize_vector(np.cross(x_axis, y_axis))
    x_axis = normalize_vector(np.cross(y_axis, z_axis))
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    return rot.astype(np.float32)


def intersect_ray_with_plane(
    u: float,
    v: float,
    plane_normal: np.ndarray,
    plane_offset: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray | None:
    ray = pixel_to_camera_ray(u, v, intrinsics)
    denom = float(np.dot(plane_normal, ray))
    if abs(denom) < 1e-6:
        return None
    depth_scale = -float(plane_offset) / denom
    if depth_scale <= 0:
        return None
    return (ray * depth_scale).astype(np.float32)


def fit_plane_svd(points_xyz: np.ndarray) -> tuple[np.ndarray, float]:
    if points_xyz.ndim != 2 or points_xyz.shape[0] < 3 or points_xyz.shape[1] != 3:
        raise ValueError("Need at least 3 points with shape (N, 3) to fit a plane")
    centroid = points_xyz.mean(axis=0)
    centered = points_xyz - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = normalize_vector(vh[-1])
    offset = -float(np.dot(normal, centroid))
    return normal, offset


def point_plane_signed_distance(points_xyz: np.ndarray, plane_normal: np.ndarray, plane_offset: float) -> np.ndarray:
    return points_xyz @ plane_normal + plane_offset


def point_plane_distance(points_xyz: np.ndarray, plane_normal: np.ndarray, plane_offset: float) -> np.ndarray:
    return np.abs(point_plane_signed_distance(points_xyz, plane_normal, plane_offset))


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("mask is empty")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def mask_centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("mask is empty")
    return np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def principal_axis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    center = pts.mean(axis=0)
    centered = pts - center
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_axis = eigvecs[:, np.argmax(eigvals)]
    return center, main_axis


def rotated_bbox(mask: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float]:
    ys, xs = np.where(mask > 0)
    contour = np.stack([xs, ys], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(contour)
    return rect


def pixel_width_profile(mask: np.ndarray) -> np.ndarray:
    h = mask.shape[0]
    profile = np.zeros(h, dtype=np.float32)
    for y in range(h):
        xs = np.where(mask[y] > 0)[0]
        if len(xs) > 0:
            profile[y] = float(xs.max() - xs.min() + 1)
    return profile


def estimate_top_width(mask: np.ndarray, band_ratio: float = 0.08) -> float:
    ys, _ = np.where(mask > 0)
    if len(ys) == 0:
        return 0.0
    y_top = int(ys.min())
    band_h = max(3, int(mask.shape[0] * band_ratio))
    band = mask[y_top : min(mask.shape[0], y_top + band_h)]
    xs = np.where(band > 0)[1]
    if len(xs) == 0:
        return 0.0
    return float(xs.max() - xs.min() + 1)


def median_valid_depth(depth: np.ndarray, mask: np.ndarray) -> float:
    values = depth[(mask > 0) & (depth > 0)]
    if values.size == 0:
        return 1000.0
    return float(np.median(values))


def project_circle_radius_to_pixels(radius: float, depth_z: float, intrinsics: CameraIntrinsics) -> float:
    return intrinsics.fx * radius / max(depth_z, 1e-6)


def metric_radius_from_pixel_radius(pixel_radius: float, depth_z: float, intrinsics: CameraIntrinsics) -> float:
    return pixel_radius * depth_z / max(intrinsics.fx, 1e-6)
