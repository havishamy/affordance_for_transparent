from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from primitive_recovery.geometry import (
    CameraIntrinsics,
    backproject_pixel_to_camera,
    bbox_from_mask,
    intersect_ray_with_plane,
    mask_centroid,
    median_valid_depth,
    metric_radius_from_pixel_radius,
    rotation_matrix_from_y_axis,
    rotation_matrix_to_rpy,
    rotated_bbox,
)
from primitive_recovery.render_beaker import render_beaker_mask
from primitive_recovery.table_plane import TablePlaneEstimate, estimate_cylinder_contact_point, estimate_table_plane_from_depth
from primitive_recovery.templates import BeakerParams, clip_beaker_params


@dataclass
class BeakerInitializationResult:
    params: BeakerParams
    table_plane: TablePlaneEstimate | None
    init_center_world: np.ndarray
    init_contact_world: np.ndarray | None
    height_px: float
    radius_px: float
    height_metric: float
    source: str


def initialize_beaker_from_mask_and_depth(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> BeakerInitializationResult:
    bbox = bbox_from_mask(mask)
    center = mask_centroid(mask)
    rect = rotated_bbox(mask)
    (_, _), (w_rect, h_rect), _ = rect

    long_side = max(w_rect, h_rect)
    short_side = min(w_rect, h_rect)
    height_px = long_side
    radius_px = short_side * 0.5

    z = median_valid_depth(depth, mask)
    center_xyz = backproject_pixel_to_camera(center[0], center[1], z, intrinsics)
    radius_metric = metric_radius_from_pixel_radius(radius_px, z, intrinsics)
    height_metric = float(height_px) * float(z) / max(float(intrinsics.fy), 1e-6)

    table_plane = estimate_table_plane_from_depth(mask, depth, intrinsics)
    roll = 0.0
    pitch = 0.0
    yaw = 0.0
    contact_world = estimate_cylinder_contact_point(mask, depth, intrinsics)
    center_world = center_xyz.copy()
    source = "mask_depth_center"

    if table_plane is not None:
        axis_world = -table_plane.normal.astype(np.float32)
        if axis_world[1] > 0:
            axis_world = -axis_world
        rot = rotation_matrix_from_y_axis(axis_world)
        roll, pitch, yaw = rotation_matrix_to_rpy(rot)

        plane_center = intersect_ray_with_plane(center[0], center[1], table_plane.normal, table_plane.offset, intrinsics)
        if plane_center is None:
            plane_center = center_xyz.copy()

        support_point = plane_center
        if contact_world is not None:
            support_point = contact_world.astype(np.float32)
        center_world = support_point - axis_world * (0.5 * height_metric)
        source = table_plane.source
    else:
        center_world = center_xyz

    params = BeakerParams(
        x=float(center_world[0]),
        y=float(center_world[1]),
        z=float(center_world[2]),
        roll=float(roll),
        pitch=float(pitch),
        yaw=float(yaw),
        radius=float(radius_metric),
        height=float(height_metric),
    )
    params = clip_beaker_params(params)

    try:
        rendered_mask = render_beaker_mask(params, mask.shape, intrinsics)
        rendered_centroid = mask_centroid(rendered_mask)
        delta_uv = center - rendered_centroid
        center_world[0] += float(delta_uv[0]) * float(params.z) / max(float(intrinsics.fx), 1e-6)
        center_world[1] += float(delta_uv[1]) * float(params.z) / max(float(intrinsics.fy), 1e-6)
        params = BeakerParams(
            x=float(center_world[0]),
            y=float(center_world[1]),
            z=float(center_world[2]),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            radius=float(radius_metric),
            height=float(height_metric),
        )
    except ValueError:
        params = params

    return BeakerInitializationResult(
        params=clip_beaker_params(params),
        table_plane=table_plane,
        init_center_world=center_world.astype(np.float32),
        init_contact_world=None if contact_world is None else contact_world.astype(np.float32),
        height_px=float(height_px),
        radius_px=float(radius_px),
        height_metric=float(height_metric),
        source=source,
    )
