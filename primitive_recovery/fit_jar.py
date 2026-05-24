from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from affordance.utils.image_ops import ensure_dir, load_mask
from primitive_recovery.geometry import CameraIntrinsics
from primitive_recovery.init_jar import initialize_jar_from_mask_and_depth
from primitive_recovery.losses import (
    axis_angle_loss,
    contour_chamfer_loss,
    mask_iou_loss,
    prior_regularization,
    robust_depth_loss,
    top_width_loss,
)
from primitive_recovery.render_jar import render_jar_depth, render_jar_mask
from primitive_recovery.templates_jar import JarParams, clip_jar_params


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


def make_objective(observed_mask: np.ndarray, observed_depth: np.ndarray, intrinsics: CameraIntrinsics, history: list[dict]):
    image_shape = observed_mask.shape

    def objective(theta: np.ndarray) -> float:
        params = JarParams.from_vector(theta.tolist())
        params = clip_jar_params(params)
        rendered_mask = render_jar_mask(params, image_shape, intrinsics)
        rendered_depth = render_jar_depth(params, image_shape, intrinsics)

        l_mask = mask_iou_loss(rendered_mask, observed_mask)
        l_depth = robust_depth_loss(rendered_depth, observed_depth, observed_mask)
        l_contour = contour_chamfer_loss(rendered_mask, observed_mask)
        l_axis = axis_angle_loss(rendered_mask, observed_mask)
        l_top = top_width_loss(rendered_mask, observed_mask)
        l_prior = prior_regularization(params.body_radius, params.body_height, params.roll, params.pitch)

        total = 4.0 * l_mask + 0.05 * l_depth + 0.5 * l_contour + 0.6 * l_axis + 0.5 * l_top + 0.1 * l_prior
        history.append(
            {
                "total": float(total),
                "mask": float(l_mask),
                "depth": float(l_depth),
                "contour": float(l_contour),
                "axis": float(l_axis),
                "top_width": float(l_top),
                "prior": float(l_prior),
                "params": {
                    "x": float(params.x),
                    "y": float(params.y),
                    "z": float(params.z),
                    "roll": float(params.roll),
                    "pitch": float(params.pitch),
                    "yaw": float(params.yaw),
                    "body_radius": float(params.body_radius),
                    "body_height": float(params.body_height),
                    "lid_radius": float(params.lid_radius),
                    "lid_height": float(params.lid_height),
                },
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


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    observed_mask = load_mask(args.mask)
    observed_mask = (observed_mask > 0.5).astype(np.uint8)
    observed_depth = load_depth(args.depth)
    intrinsics = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)

    init_params = initialize_jar_from_mask_and_depth(observed_mask, observed_depth, intrinsics)
    loss_history: list[dict] = []
    objective = make_objective(observed_mask, observed_depth, intrinsics, loss_history)

    result = minimize(
        objective,
        x0=np.asarray(init_params.to_vector(), dtype=np.float64),
        method="Powell",
        options={"maxiter": args.maxiter, "disp": True},
    )

    final_params = JarParams.from_vector(result.x.tolist())
    final_params = clip_jar_params(final_params)
    rendered_mask = render_jar_mask(final_params, observed_mask.shape, intrinsics)
    rendered_depth = render_jar_depth(final_params, observed_mask.shape, intrinsics)

    summary = {
        "success": bool(result.success),
        "message": str(result.message),
        "final_loss": float(result.fun),
        "nfev": int(result.nfev),
        "initial_params": asdict(init_params),
        "final_params": asdict(final_params),
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
