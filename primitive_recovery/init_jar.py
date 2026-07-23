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
from primitive_recovery.render_jar import render_jar_mask
from primitive_recovery.table_plane import TablePlaneEstimate, estimate_cylinder_contact_point, estimate_table_plane_from_depth
from primitive_recovery.templates_jar import JarParams, clip_jar_params


@dataclass
class JarInitializationResult:
    params: JarParams
    table_plane: TablePlaneEstimate | None
    init_center_world: np.ndarray
    init_contact_world: np.ndarray | None
    body_height_px: float
    body_radius_px: float
    body_height_metric: float
    neck_stack_height_metric: float
    source: str


def initialize_jar_from_mask_and_depth(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> JarInitializationResult:
    bbox = bbox_from_mask(mask)
    center = mask_centroid(mask)
    rect = rotated_bbox(mask)
    (_, _), (w_rect, h_rect), _ = rect

    long_side = max(w_rect, h_rect)
    short_side = min(w_rect, h_rect)
    body_height_px = long_side * 0.58
    shoulder_height_px = long_side * 0.16
    neck_height_px = long_side * 0.18
    lip_height_px = max(6.0, long_side * 0.08)
    body_radius_px = short_side * 0.5
    neck_radius_px = body_radius_px * 0.58
    lip_radius_px = body_radius_px * 0.74

    z = median_valid_depth(depth, mask)
    center_xyz = backproject_pixel_to_camera(center[0], center[1], z, intrinsics)
    body_radius_metric = metric_radius_from_pixel_radius(body_radius_px, z, intrinsics)
    neck_radius_metric = metric_radius_from_pixel_radius(neck_radius_px, z, intrinsics)
    lip_radius_metric = metric_radius_from_pixel_radius(lip_radius_px, z, intrinsics)
    body_height_metric = float(body_height_px) * float(z) / max(float(intrinsics.fy), 1e-6)
    shoulder_height_metric = float(shoulder_height_px) * float(z) / max(float(intrinsics.fy), 1e-6)
    neck_height_metric = float(neck_height_px) * float(z) / max(float(intrinsics.fy), 1e-6)
    lip_height_metric = float(lip_height_px) * float(z) / max(float(intrinsics.fy), 1e-6)

    table_plane = estimate_table_plane_from_depth(mask, depth, intrinsics)
    contact_world = estimate_cylinder_contact_point(mask, depth, intrinsics)
    center_world = center_xyz.copy()
    roll = 0.0
    pitch = 0.0
    yaw = 0.0
    source = "mask_depth_center"

    if table_plane is not None:
        axis_world = -table_plane.normal.astype(np.float32)
        if axis_world[1] > 0:
            axis_world = -axis_world
        rot = rotation_matrix_from_y_axis(axis_world)
        roll, pitch, yaw = rotation_matrix_to_rpy(rot)

        source = table_plane.source

    params = JarParams(
        x=float(center_world[0]),
        y=float(center_world[1]),
        z=float(center_world[2]),
        roll=float(roll),
        pitch=float(pitch),
        yaw=float(yaw),
        body_radius=float(body_radius_metric),
        body_height=float(body_height_metric),
        shoulder_height=float(shoulder_height_metric),
        neck_radius=float(neck_radius_metric),
        neck_height=float(neck_height_metric),
        lip_radius=float(lip_radius_metric),
        lip_height=float(lip_height_metric),
    )
    params = clip_jar_params(params)

    try:
        rendered_mask = render_jar_mask(params, mask.shape, intrinsics)
        rendered_centroid = mask_centroid(rendered_mask)
        delta_uv = center - rendered_centroid
        center_world[0] += float(delta_uv[0]) * float(params.z) / max(float(intrinsics.fx), 1e-6)
        center_world[1] += float(delta_uv[1]) * float(params.z) / max(float(intrinsics.fy), 1e-6)
        params = JarParams(
            x=float(center_world[0]),
            y=float(center_world[1]),
            z=float(center_world[2]),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            body_radius=float(body_radius_metric),
            body_height=float(body_height_metric),
            shoulder_height=float(shoulder_height_metric),
            neck_radius=float(neck_radius_metric),
            neck_height=float(neck_height_metric),
            lip_radius=float(lip_radius_metric),
            lip_height=float(lip_height_metric),
        )
    except ValueError:
        params = params

    return JarInitializationResult(
        params=clip_jar_params(params),
        table_plane=table_plane,
        init_center_world=center_world.astype(np.float32),
        init_contact_world=None if contact_world is None else contact_world.astype(np.float32),
        body_height_px=float(body_height_px),
        body_radius_px=float(body_radius_px),
        body_height_metric=float(body_height_metric),
        neck_stack_height_metric=float(shoulder_height_metric + neck_height_metric + lip_height_metric),
        source=source,
    )
