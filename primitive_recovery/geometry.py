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


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("mask is empty")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


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

