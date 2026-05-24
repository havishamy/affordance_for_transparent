from __future__ import annotations

import numpy as np

from primitive_recovery.geometry import (
    CameraIntrinsics,
    backproject_pixel_to_camera,
    bbox_from_mask,
    median_valid_depth,
    metric_radius_from_pixel_radius,
    principal_axis,
    rotated_bbox,
)
from primitive_recovery.templates_jar import JarParams, clip_jar_params


def initialize_jar_from_mask_and_depth(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> JarParams:
    bbox = bbox_from_mask(mask)
    center, _ = principal_axis(mask)
    rect = rotated_bbox(mask)
    (_, _), (w_rect, h_rect), _ = rect

    long_side = max(w_rect, h_rect)
    short_side = min(w_rect, h_rect)
    body_height_px = long_side * 0.75
    lid_height_px = max(8.0, long_side * 0.15)
    body_radius_px = short_side * 0.5
    lid_radius_px = body_radius_px * 0.98

    z = median_valid_depth(depth, mask)
    xyz = backproject_pixel_to_camera(center[0], center[1], z, intrinsics)
    body_radius_metric = metric_radius_from_pixel_radius(body_radius_px, z, intrinsics)
    lid_radius_metric = metric_radius_from_pixel_radius(lid_radius_px, z, intrinsics)

    params = JarParams(
        x=float(xyz[0]),
        y=float(xyz[1]),
        z=float(xyz[2]),
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        body_radius=float(body_radius_metric),
        body_height=float(body_height_px),
        lid_radius=float(lid_radius_metric),
        lid_height=float(lid_height_px),
    )
    return clip_jar_params(params)
