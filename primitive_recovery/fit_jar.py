from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from affordance.utils.image_ops import ensure_dir, load_mask
from primitive_recovery.geometry import CameraIntrinsics, intersect_ray_with_plane, project_camera_to_pixel
from primitive_recovery.init_jar import JarInitializationResult, initialize_jar_from_mask_and_depth
from primitive_recovery.losses import (
    axis_alignment_loss,
    axis_angle_loss,
    bottom_alignment_loss,
    centroid_alignment_loss,
    contour_chamfer_loss,
    mask_iou_loss,
    plane_contact_loss,
    prior_regularization,
    robust_depth_loss,
    top_width_loss,
)
from primitive_recovery.render_jar import render_jar_depth, render_jar_mask
from primitive_recovery.table_plane import TablePlaneEstimate
from primitive_recovery.templates_jar import JarParams, clip_jar_params


def jar_shape_prior(params: JarParams) -> float:
    penalties = 0.0
    total_height = params.body_height + params.shoulder_height + params.neck_height + params.lip_height
    total_ratio = total_height / max(params.body_radius, 1e-6)
    if total_ratio < 2.0:
        penalties += (2.0 - total_ratio) * 30.0
    if total_ratio > 5.0:
        penalties += (total_ratio - 5.0) * 25.0

    neck_ratio = params.neck_radius / max(params.body_radius, 1e-6)
    if neck_ratio < 0.35:
        penalties += (0.35 - neck_ratio) * 80.0
    if neck_ratio > 0.8:
        penalties += (neck_ratio - 0.8) * 80.0

    lip_ratio = params.lip_radius / max(params.neck_radius, 1e-6)
    if lip_ratio < 1.05:
        penalties += (1.05 - lip_ratio) * 60.0
    if lip_ratio > 1.5:
        penalties += (lip_ratio - 1.5) * 60.0

    shoulder_ratio = params.shoulder_height / max(total_height, 1e-6)
    if shoulder_ratio < 0.08:
        penalties += (0.08 - shoulder_ratio) * 100.0
    if shoulder_ratio > 0.28:
        penalties += (shoulder_ratio - 0.28) * 100.0
    return float(penalties)


def pack_optimization_vector(params: JarParams) -> np.ndarray:
    return np.asarray(
        [
            params.x,
            params.y,
            params.z,
            params.body_radius,
            params.body_height,
            params.shoulder_height,
            params.neck_radius,
            params.neck_height,
            params.lip_radius,
            params.lip_height,
        ],
        dtype=np.float64,
    )


def unpack_optimization_vector(theta: np.ndarray, fixed_pose: JarParams) -> JarParams:
    x, y, z, body_radius, body_height, shoulder_height, neck_radius, neck_height, lip_radius, lip_height = theta.tolist()
    params = JarParams(
        x=float(x),
        y=float(y),
        z=float(z),
        roll=float(fixed_pose.roll),
        pitch=float(fixed_pose.pitch),
        yaw=float(fixed_pose.yaw),
        body_radius=float(body_radius),
        body_height=float(body_height),
        shoulder_height=float(shoulder_height),
        neck_radius=float(neck_radius),
        neck_height=float(neck_height),
        lip_radius=float(lip_radius),
        lip_height=float(lip_height),
    )
    return clip_jar_params(params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a simple lidded jar primitive from mask + depth.")
    parser.add_argument("--mask", type=str, required=True, help="Path to binary object mask.")
    parser.add_argument("--depth", type=str, required=True, help="Path to depth image.")
    parser.add_argument("--output-dir", type=str, default="primitive_outputs/jar_fit")
    parser.add_argument("--fx", type=float, default=605.74)
    parser.add_argument("--fy", type=float, default=605.42)
    parser.add_argument("--cx", type=float, default=335.28)
    parser.add_argument("--cy", type=float, default=249.04)
    parser.add_argument("--maxiter", type=int, default=150)
    return parser.parse_args()


def load_depth(path: str | Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Unable to read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def make_depth_canvas(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0
    out = np.zeros_like(depth, dtype=np.uint8)
    if np.any(valid):
        values = depth[valid]
        lo, hi = float(values.min()), float(values.max())
        if hi > lo:
            out = np.clip((depth - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        else:
            out[valid] = 180
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def estimate_plane_anchor_pixel(
    table_plane: TablePlaneEstimate,
    intrinsics: CameraIntrinsics,
) -> tuple[tuple[int, int], np.ndarray] | None:
    ys, xs = np.where(table_plane.inlier_mask > 0)
    if len(xs) == 0:
        return None
    u = float(np.median(xs))
    v = float(np.median(ys))
    anchor_world = intersect_ray_with_plane(u, v, table_plane.normal, table_plane.offset, intrinsics)
    if anchor_world is None:
        return None
    return (int(round(u)), int(round(v))), anchor_world


def draw_table_normal_visualization(
    canvas_bgr: np.ndarray,
    table_plane: TablePlaneEstimate,
    intrinsics: CameraIntrinsics,
    arrow_len_mm: float = 80.0,
) -> tuple[np.ndarray, dict] | tuple[None, None]:
    anchor = estimate_plane_anchor_pixel(table_plane, intrinsics)
    if anchor is None:
        return None, None

    (u0, v0), anchor_world = anchor
    normal_world = table_plane.normal.astype(np.float32)
    axis_world = -normal_world
    end_normal_world = anchor_world + normal_world * float(arrow_len_mm)
    end_axis_world = anchor_world + axis_world * float(arrow_len_mm)
    end_normal_uv = project_camera_to_pixel(end_normal_world, intrinsics)
    end_axis_uv = project_camera_to_pixel(end_axis_world, intrinsics)
    if end_normal_uv is None or end_axis_uv is None:
        return None, None

    vis = canvas_bgr.copy()
    overlay = vis.copy()
    overlay[table_plane.ring_mask > 0] = (60, 60, 60)
    overlay[table_plane.inlier_mask > 0] = (0, 180, 255)
    vis = cv2.addWeighted(vis, 0.72, overlay, 0.28, 0.0)

    p0 = (int(round(u0)), int(round(v0)))
    p_normal = (int(round(end_normal_uv[0])), int(round(end_normal_uv[1])))
    p_axis = (int(round(end_axis_uv[0])), int(round(end_axis_uv[1])))
    cv2.circle(vis, p0, 5, (255, 255, 255), -1)
    cv2.arrowedLine(vis, p0, p_normal, (0, 0, 255), 3, tipLength=0.18)
    cv2.arrowedLine(vis, p0, p_axis, (0, 255, 0), 3, tipLength=0.18)
    cv2.putText(vis, "table n", (p_normal[0] + 6, p_normal[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "jar axis -n", (p_axis[0] + 6, p_axis[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    meta = {
        "anchor_pixel": [int(p0[0]), int(p0[1])],
        "normal_tip_pixel": [int(p_normal[0]), int(p_normal[1])],
        "axis_tip_pixel": [int(p_axis[0]), int(p_axis[1])],
        "arrow_length_mm": float(arrow_len_mm),
    }
    return vis, meta


def make_objective(
    observed_mask: np.ndarray,
    observed_depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    history: list[dict],
    fixed_pose: JarParams,
    table_plane: TablePlaneEstimate | None = None,
):
    image_shape = observed_mask.shape
    table_axis = None if table_plane is None else -table_plane.normal.astype(np.float32)

    def objective(theta: np.ndarray) -> float:
        params = unpack_optimization_vector(theta, fixed_pose)
        rendered_mask = render_jar_mask(params, image_shape, intrinsics)
        rendered_depth = render_jar_depth(params, image_shape, intrinsics)

        l_mask = mask_iou_loss(rendered_mask, observed_mask)
        l_depth = robust_depth_loss(rendered_depth, observed_depth, observed_mask)
        l_contour = contour_chamfer_loss(rendered_mask, observed_mask)
        l_axis = axis_angle_loss(rendered_mask, observed_mask)
        l_centroid = centroid_alignment_loss(rendered_mask, observed_mask)
        l_bottom = bottom_alignment_loss(rendered_mask, observed_mask)
        l_top = top_width_loss(rendered_mask, observed_mask)
        l_prior = prior_regularization(params.body_radius, params.body_height, params.roll, params.pitch)
        l_shape = jar_shape_prior(params)
        l_table_axis = 0.0
        l_table_contact = 0.0
        if table_plane is not None and table_axis is not None:
            l_table_axis = axis_alignment_loss(params.roll, params.pitch, params.yaw, table_axis)
            center_xyz = np.asarray([params.x, params.y, params.z], dtype=np.float32)
            total_height = params.body_height + params.shoulder_height + params.neck_height + params.lip_height
            l_table_contact = plane_contact_loss(
                center_xyz=center_xyz,
                height=total_height,
                roll=params.roll,
                pitch=params.pitch,
                yaw=params.yaw,
                plane_normal=table_plane.normal,
                plane_offset=table_plane.offset,
            )

        total = (
            6.0 * l_mask
            + 0.003 * l_depth
            + 0.8 * l_contour
            + 0.2 * l_axis
            + 0.08 * l_centroid
            + 0.12 * l_bottom
            + 0.5 * l_top
            + 0.1 * l_prior
            + 0.2 * l_shape
            + 1.2 * l_table_axis
            + 0.12 * l_table_contact
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
                "shape": float(l_shape),
                "table_axis": float(l_table_axis),
                "table_contact": float(l_table_contact),
                "params": {
                    "x": float(params.x),
                    "y": float(params.y),
                    "z": float(params.z),
                    "roll": float(params.roll),
                    "pitch": float(params.pitch),
                    "yaw": float(params.yaw),
                    "body_radius": float(params.body_radius),
                    "body_height": float(params.body_height),
                    "shoulder_height": float(params.shoulder_height),
                    "neck_radius": float(params.neck_radius),
                    "neck_height": float(params.neck_height),
                    "lip_radius": float(params.lip_radius),
                    "lip_height": float(params.lip_height),
                },
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
    final_params: JarParams,
    final_mask_iou: float,
    table_plane: TablePlaneEstimate | None = None,
) -> dict:
    optimizer_success = len(history) > 0

    loss_stable = False
    if len(history) >= 10:
        last_vals = [h["total"] for h in history[-10:]]
        diffs = [abs(last_vals[i + 1] - last_vals[i]) for i in range(len(last_vals) - 1)]
        loss_stable = max(diffs) < 1e-2

    parameters_reasonable = True
    total_height = final_params.body_height + final_params.shoulder_height + final_params.neck_height + final_params.lip_height
    if final_params.body_radius <= 0 or total_height <= 0 or final_params.z <= 0:
        parameters_reasonable = False
    ratio = total_height / max(final_params.body_radius, 1e-6)
    if ratio < 1.0 or ratio > 8.0:
        parameters_reasonable = False

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
        "final_height_radius_ratio": float(ratio),
        "table_axis_error": None if table_axis_error is None else float(table_axis_error),
    }


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    observed_mask = load_mask(args.mask)
    observed_mask = (observed_mask > 0.5).astype(np.uint8)
    observed_depth = load_depth(args.depth)
    intrinsics = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)

    init_result: JarInitializationResult = initialize_jar_from_mask_and_depth(observed_mask, observed_depth, intrinsics)
    init_params = init_result.params
    loss_history: list[dict] = []
    objective = make_objective(observed_mask, observed_depth, intrinsics, loss_history, init_params, init_result.table_plane)

    result = minimize(
        objective,
        x0=pack_optimization_vector(init_params),
        method="Powell",
        options={"maxiter": args.maxiter, "disp": True},
    )

    final_params = unpack_optimization_vector(result.x, init_params)
    rendered_mask = render_jar_mask(final_params, observed_mask.shape, intrinsics)
    rendered_depth = render_jar_depth(final_params, observed_mask.shape, intrinsics)
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
        "initialization": {
            "source": init_result.source,
            "pose_locked_during_optimization": True,
            "body_height_px": float(init_result.body_height_px),
            "body_radius_px": float(init_result.body_radius_px),
            "body_height_metric": float(init_result.body_height_metric),
            "neck_stack_height_metric": float(init_result.neck_stack_height_metric),
            "init_center_world": init_result.init_center_world.tolist(),
            "init_contact_world": None if init_result.init_contact_world is None else init_result.init_contact_world.tolist(),
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

    base_canvas = make_depth_canvas(observed_depth)
    if init_result.table_plane is not None:
        table_vis, table_vis_meta = draw_table_normal_visualization(base_canvas, init_result.table_plane, intrinsics)
        if table_vis is not None and table_vis_meta is not None:
            cv2.imwrite(str(Path(out_dir) / "table_normal_overlay.png"), table_vis)
            summary["table_plane_visualization"] = table_vis_meta

    (Path(out_dir) / "fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (Path(out_dir) / "loss_history.json").write_text(json.dumps(loss_history, indent=2), encoding="utf-8")
    cv2.imwrite(str(Path(out_dir) / "observed_mask.png"), (observed_mask * 255).astype(np.uint8))
    cv2.imwrite(str(Path(out_dir) / "rendered_mask.png"), (rendered_mask * 255).astype(np.uint8))
    cv2.imwrite(str(Path(out_dir) / "mask_overlay.png"), cv2.cvtColor(overlay_masks(observed_mask, rendered_mask), cv2.COLOR_RGB2BGR))
    if init_result.table_plane is not None:
        cv2.imwrite(str(Path(out_dir) / "table_plane_inliers.png"), (init_result.table_plane.inlier_mask * 255).astype(np.uint8))
        cv2.imwrite(str(Path(out_dir) / "table_plane_ring.png"), (init_result.table_plane.ring_mask * 255).astype(np.uint8))
    save_depth_visual(Path(out_dir) / "observed_depth.png", observed_depth, observed_mask)
    save_depth_visual(Path(out_dir) / "rendered_depth.png", rendered_depth, rendered_mask)
    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
