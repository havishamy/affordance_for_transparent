from __future__ import annotations

import argparse
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from affordance.utils.image_ops import ensure_dir, load_mask
from primitive_recovery.geometry import (
    CameraIntrinsics,
    backproject_pixel_to_camera,
    bbox_from_mask,
    mask_centroid,
    normalize_vector,
    project_camera_to_pixel,
    rotation_matrix_from_y_axis,
    rotation_matrix_to_rpy,
    rpy_to_rotation_matrix,
)
from primitive_recovery.losses import (
    axis_alignment_loss,
    axis_angle_loss,
    bottom_alignment_loss,
    centroid_alignment_loss,
    contour_chamfer_loss,
    mask_iou_loss,
    prior_regularization,
    robust_depth_loss,
    top_width_loss,
)
from primitive_recovery.table_plane import TablePlaneEstimate, estimate_table_plane_from_depth


@dataclass
class PointCloudJarParams:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    scale: float


@dataclass
class PointCloudJarInitializationResult:
    params: PointCloudJarParams
    table_plane: TablePlaneEstimate | None
    init_center_world: np.ndarray
    model_axis_local: np.ndarray
    model_extent_local: np.ndarray
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a jar point cloud with global scale + pose from mask/depth.")
    parser.add_argument("--ply", type=str, required=True, help="Path to object point cloud in PLY format.")
    parser.add_argument("--mask", type=str, required=True, help="Path to binary object mask.")
    parser.add_argument("--depth", type=str, required=True, help="Path to depth image.")
    parser.add_argument("--rgb", type=str, default=None, help="Optional RGB image for visualization.")
    parser.add_argument("--output-dir", type=str, default="primitive_outputs/jar_fit_pointcloud")
    parser.add_argument("--fx", type=float, default=605.74)
    parser.add_argument("--fy", type=float, default=605.42)
    parser.add_argument("--cx", type=float, default=335.28)
    parser.add_argument("--cy", type=float, default=249.04)
    parser.add_argument("--maxiter", type=int, default=120)
    return parser.parse_args()


def load_depth(path: str | Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Unable to read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def load_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_ply_vertices(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PLY: missing end_header")
            decoded = line.decode("ascii", errors="ignore").strip()
            header.append(decoded)
            if decoded == "end_header":
                break

        if header[0] != "ply":
            raise ValueError("Invalid PLY: missing ply magic")
        if "format binary_little_endian 1.0" not in header:
            raise ValueError("Only binary_little_endian PLY is supported")

        vertex_count = None
        for line in header:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                break
        if vertex_count is None:
            raise ValueError("PLY does not contain vertex count")

        vertices = np.fromfile(f, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]), count=vertex_count)
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    return points


def canonicalize_point_cloud(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Point cloud must have shape (N, 3)")
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    axis0 = normalize_vector(eigvecs[:, 0])
    axis1 = normalize_vector(eigvecs[:, 1])
    axis2 = normalize_vector(np.cross(axis0, axis1))

    # Prefer the longest extent to be the vertical local y-axis.
    projected = centered @ np.stack([axis0, axis1, axis2], axis=1)
    extents = projected.max(axis=0) - projected.min(axis=0)
    vertical_idx = int(np.argmax(extents))
    basis = [axis0, axis1, axis2]
    y_axis = basis[vertical_idx]
    remaining = [basis[i] for i in range(3) if i != vertical_idx]
    x_axis = remaining[0]
    x_axis = normalize_vector(x_axis - np.dot(x_axis, y_axis) * y_axis)
    z_axis = normalize_vector(np.cross(x_axis, y_axis))
    x_axis = normalize_vector(np.cross(y_axis, z_axis))
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    canonical = centered @ rot
    extent = canonical.max(axis=0) - canonical.min(axis=0)
    return canonical.astype(np.float32), rot.astype(np.float32), extent.astype(np.float32)


def transform_points(points_local: np.ndarray, params: PointCloudJarParams) -> np.ndarray:
    rot = rpy_to_rotation_matrix(params.roll, params.pitch, params.yaw)
    pts = (rot @ (points_local * params.scale).T).T
    pts += np.asarray([params.x, params.y, params.z], dtype=np.float32)
    return pts


def render_point_cloud_mask(
    points_local: np.ndarray,
    params: PointCloudJarParams,
    image_shape: tuple[int, int],
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pts_world = transform_points(points_local, params)
    pixels = []
    for p in pts_world:
        uv = project_camera_to_pixel(p, intrinsics)
        if uv is not None:
            pixels.append(uv)
    if not pixels:
        return mask
    pix = np.asarray(pixels, dtype=np.float32)
    pix_i = np.round(pix).astype(np.int32)
    pix_i[:, 0] = np.clip(pix_i[:, 0], 0, w - 1)
    pix_i[:, 1] = np.clip(pix_i[:, 1], 0, h - 1)
    hull = cv2.convexHull(pix_i.reshape(-1, 1, 2))
    cv2.fillConvexPoly(mask, hull[:, 0, :], 1)
    return mask


def render_point_cloud_depth(
    points_local: np.ndarray,
    params: PointCloudJarParams,
    image_shape: tuple[int, int],
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    h, w = image_shape
    depth = np.zeros((h, w), dtype=np.float32)
    pts_world = transform_points(points_local, params)
    for p in pts_world:
        uv = project_camera_to_pixel(p, intrinsics)
        if uv is None:
            continue
        u = int(round(uv[0]))
        v = int(round(uv[1]))
        if not (0 <= u < w and 0 <= v < h):
            continue
        z = float(p[2])
        current = depth[v, u]
        if current == 0 or z < current:
            depth[v, u] = z

    mask = render_point_cloud_mask(points_local, params, image_shape, intrinsics)
    valid_u8 = (depth > 0).astype(np.uint8)
    if valid_u8.sum() > 0:
        _, labels = cv2.distanceTransformWithLabels(1 - valid_u8, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        values = depth[valid_u8 > 0]
        dense = np.zeros_like(depth)
        for y in range(h):
            for x in range(w):
                if mask[y, x] == 0:
                    continue
                if depth[y, x] > 0:
                    dense[y, x] = depth[y, x]
                else:
                    idx = labels[y, x] - 1
                    if 0 <= idx < len(values):
                        dense[y, x] = values[idx]
        depth = dense
    depth *= mask.astype(np.float32)
    return depth


def initialize_from_point_cloud_and_mask(
    points_local: np.ndarray,
    model_extent: np.ndarray,
    model_axis_local: np.ndarray,
    observed_mask: np.ndarray,
    observed_depth: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> PointCloudJarInitializationResult:
    center_uv = mask_centroid(observed_mask)
    z = float(np.median(observed_depth[(observed_mask > 0) & (observed_depth > 0)]))
    if not np.isfinite(z) or z <= 0:
        z = 1000.0
    center_world = backproject_pixel_to_camera(center_uv[0], center_uv[1], z, intrinsics)
    bbox = bbox_from_mask(observed_mask)
    x1, y1, x2, y2 = bbox
    mask_h = float(y2 - y1)
    base_scale = mask_h * z / max(float(intrinsics.fy), 1e-6) / max(float(model_extent[1]), 1e-6)

    table_plane = estimate_table_plane_from_depth(observed_mask, observed_depth, intrinsics)
    roll = 0.0
    pitch = 0.0
    yaw = 0.0
    source = "mask_depth_center"
    if table_plane is not None:
        target_axis = -table_plane.normal.astype(np.float32)
        if target_axis[1] > 0:
            target_axis = -target_axis
        x_hint = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        rot = rotation_matrix_from_y_axis(target_axis, x_hint=x_hint)
        roll, pitch, yaw = rotation_matrix_to_rpy(rot)
        source = table_plane.source

    params = PointCloudJarParams(
        x=float(center_world[0]),
        y=float(center_world[1]),
        z=float(center_world[2]),
        roll=float(roll),
        pitch=float(pitch),
        yaw=float(yaw),
        scale=float(max(base_scale, 1e-3)),
    )

    try:
        rendered_mask = render_point_cloud_mask(points_local, params, observed_mask.shape, intrinsics)
        rendered_centroid = mask_centroid(rendered_mask)
        delta_uv = center_uv - rendered_centroid
        center_world[0] += float(delta_uv[0]) * float(params.z) / max(float(intrinsics.fx), 1e-6)
        center_world[1] += float(delta_uv[1]) * float(params.z) / max(float(intrinsics.fy), 1e-6)
        params = PointCloudJarParams(
            x=float(center_world[0]),
            y=float(center_world[1]),
            z=float(center_world[2]),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            scale=float(max(base_scale, 1e-3)),
        )
    except ValueError:
        pass

    return PointCloudJarInitializationResult(
        params=params,
        table_plane=table_plane,
        init_center_world=center_world.astype(np.float32),
        model_axis_local=model_axis_local.astype(np.float32),
        model_extent_local=model_extent.astype(np.float32),
        source=source,
    )


def pack_optimization_vector(params: PointCloudJarParams) -> np.ndarray:
    return np.asarray([params.x, params.y, params.z, params.scale], dtype=np.float64)


def unpack_optimization_vector(theta: np.ndarray, fixed_pose: PointCloudJarParams) -> PointCloudJarParams:
    x, y, z, scale = theta.tolist()
    return PointCloudJarParams(
        x=float(x),
        y=float(y),
        z=max(float(z), 10.0),
        roll=float(fixed_pose.roll),
        pitch=float(fixed_pose.pitch),
        yaw=float(fixed_pose.yaw),
        scale=max(float(scale), 1e-3),
    )


def pointcloud_scale_prior(params: PointCloudJarParams) -> float:
    penalties = 0.0
    if params.scale <= 0:
        penalties += 1000.0
    return float(penalties)


def make_objective(
    points_local: np.ndarray,
    observed_mask: np.ndarray,
    observed_depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    history: list[dict],
    fixed_pose: PointCloudJarParams,
    table_plane: TablePlaneEstimate | None = None,
) -> callable:
    image_shape = observed_mask.shape
    table_axis = None if table_plane is None else -table_plane.normal.astype(np.float32)

    def objective(theta: np.ndarray) -> float:
        params = unpack_optimization_vector(theta, fixed_pose)
        rendered_mask = render_point_cloud_mask(points_local, params, image_shape, intrinsics)
        rendered_depth = render_point_cloud_depth(points_local, params, image_shape, intrinsics)

        l_mask = mask_iou_loss(rendered_mask, observed_mask)
        l_depth = robust_depth_loss(rendered_depth, observed_depth, observed_mask)
        l_contour = contour_chamfer_loss(rendered_mask, observed_mask)
        l_axis = axis_angle_loss(rendered_mask, observed_mask)
        l_centroid = centroid_alignment_loss(rendered_mask, observed_mask)
        l_bottom = bottom_alignment_loss(rendered_mask, observed_mask)
        l_top = top_width_loss(rendered_mask, observed_mask)
        l_prior = pointcloud_scale_prior(params)
        l_table_axis = 0.0
        if table_plane is not None and table_axis is not None:
            l_table_axis = axis_alignment_loss(params.roll, params.pitch, params.yaw, table_axis)

        total = (
            6.0 * l_mask
            + 0.003 * l_depth
            + 0.8 * l_contour
            + 0.2 * l_axis
            + 0.08 * l_centroid
            + 0.12 * l_bottom
            + 0.5 * l_top
            + 0.1 * l_prior
            + 1.2 * l_table_axis
        )
        history.append(
            {
                "total": float(total),
                "mask": float(l_mask),
                "depth": float(l_depth),
                "contour": float(l_contour),
                "axis": float(l_axis),
                "centroid": float(l_centroid),
                "bottom": float(l_bottom),
                "top_width": float(l_top),
                "prior": float(l_prior),
                "table_axis": float(l_table_axis),
                "params": asdict(params),
                "pose_locked": True,
            }
        )
        return float(total)

    return objective


def overlay_masks(observed_mask: np.ndarray, rendered_mask: np.ndarray) -> np.ndarray:
    h, w = observed_mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[observed_mask > 0] = (0, 255, 0)
    canvas[rendered_mask > 0] = (255, 0, 0)
    overlap = (observed_mask > 0) & (rendered_mask > 0)
    canvas[overlap] = (255, 255, 0)
    return canvas


def save_depth_visual(path: Path, depth: np.ndarray, mask: np.ndarray | None = None) -> None:
    vis = depth.copy()
    if mask is not None:
        vis = vis * (mask > 0)
    valid = vis[vis > 0]
    out = np.zeros_like(vis, dtype=np.uint8)
    if valid.size > 0:
        lo, hi = float(valid.min()), float(valid.max())
        if hi > lo:
            out = np.clip((vis - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), out)


def assess_convergence(
    history: list[dict],
    final_params: PointCloudJarParams,
    final_mask_iou: float,
    table_plane: TablePlaneEstimate | None = None,
) -> dict:
    optimizer_success = len(history) > 0
    loss_stable = False
    if len(history) >= 10:
        last_vals = [h["total"] for h in history[-10:]]
        diffs = [abs(last_vals[i + 1] - last_vals[i]) for i in range(len(last_vals) - 1)]
        loss_stable = max(diffs) < 1e-2

    parameters_reasonable = final_params.scale > 0 and final_params.z > 0
    table_axis_error = None
    if table_plane is not None:
        table_axis = -table_plane.normal.astype(np.float32)
        table_axis_error = axis_alignment_loss(final_params.roll, final_params.pitch, final_params.yaw, table_axis)
        if table_axis_error > 0.15:
            parameters_reasonable = False

    visual_fit_reasonable = final_mask_iou >= 0.55
    overall = optimizer_success and parameters_reasonable and visual_fit_reasonable
    return {
        "optimizer_success": optimizer_success,
        "loss_stable": bool(loss_stable),
        "parameters_reasonable": bool(parameters_reasonable),
        "visual_fit_reasonable": bool(visual_fit_reasonable),
        "overall_trustworthy": bool(overall),
        "final_mask_iou": float(final_mask_iou),
        "table_axis_error": None if table_axis_error is None else float(table_axis_error),
    }


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    observed_mask = load_mask(args.mask)
    observed_mask = (observed_mask > 0.5).astype(np.uint8)
    observed_depth = load_depth(args.depth)
    intrinsics = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)

    points_raw = load_ply_vertices(args.ply)
    points_local, model_rot, model_extent = canonicalize_point_cloud(points_raw)
    model_axis_local = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    init_result = initialize_from_point_cloud_and_mask(points_local, model_extent, model_axis_local, observed_mask, observed_depth, intrinsics)
    init_params = init_result.params

    loss_history: list[dict] = []
    objective = make_objective(points_local, observed_mask, observed_depth, intrinsics, loss_history, init_params, init_result.table_plane)
    result = minimize(
        objective,
        x0=pack_optimization_vector(init_params),
        method="Powell",
        options={"maxiter": args.maxiter, "disp": True},
    )

    final_params = unpack_optimization_vector(result.x, init_params)
    rendered_mask = render_point_cloud_mask(points_local, final_params, observed_mask.shape, intrinsics)
    rendered_depth = render_point_cloud_depth(points_local, final_params, observed_mask.shape, intrinsics)
    final_mask_iou = 1.0 - mask_iou_loss(rendered_mask, observed_mask)
    convergence = assess_convergence(loss_history, final_params, final_mask_iou, init_result.table_plane)

    summary = {
        "success": bool(result.success),
        "message": str(result.message),
        "final_loss": float(result.fun),
        "nfev": int(result.nfev),
        "initial_params": asdict(init_params),
        "final_params": asdict(final_params),
        "convergence": convergence,
        "point_cloud_model": {
            "num_points": int(points_local.shape[0]),
            "extent_local": model_extent.tolist(),
        },
        "initialization": {
            "source": init_result.source,
            "pose_locked_during_optimization": True,
            "init_center_world": init_result.init_center_world.tolist(),
        },
        "table_plane": None
        if init_result.table_plane is None
        else {
            "source": init_result.table_plane.source,
            "normal": init_result.table_plane.normal.tolist(),
            "offset": float(init_result.table_plane.offset),
            "point_count": int(init_result.table_plane.point_count),
            "rms_error": float(init_result.table_plane.rms_error),
        },
    }
    (Path(out_dir) / "fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (Path(out_dir) / "loss_history.json").write_text(json.dumps(loss_history, indent=2), encoding="utf-8")
    cv2.imwrite(str(Path(out_dir) / "observed_mask.png"), (observed_mask * 255).astype(np.uint8))
    cv2.imwrite(str(Path(out_dir) / "rendered_mask.png"), (rendered_mask * 255).astype(np.uint8))
    cv2.imwrite(str(Path(out_dir) / "mask_overlay.png"), cv2.cvtColor(overlay_masks(observed_mask, rendered_mask), cv2.COLOR_RGB2BGR))
    save_depth_visual(Path(out_dir) / "observed_depth.png", observed_depth, observed_mask)
    save_depth_visual(Path(out_dir) / "rendered_depth.png", rendered_depth, rendered_mask)
    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
