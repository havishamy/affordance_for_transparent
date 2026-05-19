from __future__ import annotations

import math

import numpy as np

from primitive_recovery.geometry import (
    CameraIntrinsics,
    bbox_from_mask,
    median_valid_depth,
    metric_radius_from_pixel_radius,
    principal_axis,
    rotated_bbox,
)
from primitive_recovery.templates import BeakerParams, clip_beaker_params


def initialize_beaker_from_mask_and_depth(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> BeakerParams:
    bbox = bbox_from_mask(mask)
    x1, y1, x2, y2 = bbox
    center, axis = principal_axis(mask)
    rect = rotated_bbox(mask)
    (_, _), (w_rect, h_rect), angle_deg = rect

    long_side = max(w_rect, h_rect)
    short_side = min(w_rect, h_rect)
    height_px = long_side
    radius_px = short_side * 0.5

    z = median_valid_depth(depth, mask)
    radius_metric = metric_radius_from_pixel_radius(radius_px, z, intrinsics)
    yaw = math.radians(angle_deg)

    params = BeakerParams(
        cx=float(center[0]),
        cy=float(center[1]),
        z=float(z),
        radius=float(radius_metric),
        height=float(height_px),
        yaw=float(yaw),
    )
    return clip_beaker_params(params, mask.shape[1], mask.shape[0])

