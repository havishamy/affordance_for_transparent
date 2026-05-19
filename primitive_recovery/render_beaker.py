from __future__ import annotations

import math

import cv2
import numpy as np

from primitive_recovery.geometry import CameraIntrinsics, project_circle_radius_to_pixels
from primitive_recovery.templates import BeakerParams


def render_beaker_mask(
    params: BeakerParams,
    image_shape: tuple[int, int],
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    radius_px = project_circle_radius_to_pixels(params.radius, params.z, intrinsics)
    top_center = (params.cx, params.cy - 0.5 * params.height)
    bottom_center = (params.cx, params.cy + 0.5 * params.height)

    rect_w = max(2, int(round(2.0 * radius_px)))
    rect_h = max(2, int(round(params.height)))
    box = ((float(params.cx), float(params.cy)), (float(rect_w), float(rect_h)), float(np.degrees(params.yaw)))
    pts = cv2.boxPoints(box).astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 1)

    rim_thickness = max(2, int(round(radius_px * 0.12)))
    top_ellipse_axes = (max(2, int(round(radius_px))), max(2, rim_thickness))
    bottom_ellipse_axes = (max(2, int(round(radius_px))), max(2, rim_thickness))

    cv2.ellipse(
        mask,
        (int(round(top_center[0])), int(round(top_center[1]))),
        top_ellipse_axes,
        angle=np.degrees(params.yaw),
        startAngle=0,
        endAngle=360,
        color=1,
        thickness=-1,
    )
    cv2.ellipse(
        mask,
        (int(round(bottom_center[0])), int(round(bottom_center[1]))),
        bottom_ellipse_axes,
        angle=np.degrees(params.yaw),
        startAngle=0,
        endAngle=360,
        color=1,
        thickness=-1,
    )
    return mask


def render_beaker_depth(
    params: BeakerParams,
    image_shape: tuple[int, int],
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    height, width = image_shape
    mask = render_beaker_mask(params, image_shape, intrinsics)
    depth = np.zeros((height, width), dtype=np.float32)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return depth

    cx = params.cx
    radius_px = project_circle_radius_to_pixels(params.radius, params.z, intrinsics)
    if radius_px < 1e-3:
        depth[mask > 0] = params.z
        return depth

    norm_x = (xs.astype(np.float32) - cx) / radius_px
    profile = np.clip(1.0 - norm_x**2, 0.0, 1.0)
    # front-facing cylindrical shell approximation
    z_offset = params.radius * (1.0 - np.sqrt(profile + 1e-8))
    values = params.z - z_offset
    depth[ys, xs] = values
    return depth

