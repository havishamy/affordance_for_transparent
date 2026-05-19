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
from primitive_recovery.init_beaker import initialize_beaker_from_mask_and_depth
from primitive_recovery.losses import (
    axis_angle_loss,
    contour_chamfer_loss,
    mask_iou_loss,
    prior_regularization,
    robust_depth_loss,
    top_width_loss,
)
from primitive_recovery.render_beaker import render_beaker_depth, render_beaker_mask
from primitive_recovery.templates import BeakerParams, clip_beaker_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a simple beaker primitive from mask + depth.")
    parser.add_argument("--mask", type=str, default=None, help="Path to binary object mask.")
    parser.add_argument("--depth", type=str, required=True, help="Path to depth image.")
    parser.add_argument("--rgb", type=str, default=None, help="Optional RGB image path for visualization or interactive ROI.")
    parser.add_argument(
        "--bbox",
        type=int,
        nargs=4,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional bounding box to synthesize a coarse mask when no mask file is available.",
    )
    parser.add_argument(
        "--interactive-roi",
        action="store_true",
        default=False,
        help="Interactively select a ROI on the RGB image to generate a coarse mask.",
    )
    parser.add_argument("--output-dir", type=str, default="primitive_outputs/beaker_fit")
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


def make_mask_from_bbox(image_shape: tuple[int, int], bbox: list[int]) -> np.ndarray:
    h, w = image_shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = min(max(x1, 0), w - 1)
    x2 = min(max(x2, x1 + 1), w)
    y1 = min(max(y1, 0), h - 1)
    y2 = min(max(y2, y1 + 1), h)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def select_roi_bbox(rgb: np.ndarray) -> list[int]:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    x, y, w, h = cv2.selectROI("Select Beaker ROI", bgr, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Beaker ROI")
    if w <= 0 or h <= 0:
        raise ValueError("ROI selection cancelled")
    return [int(x), int(y), int(x + w), int(y + h)]


def make_objective(
    observed_mask: np.ndarray,
    observed_depth: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> callable:
    image_shape = observed_mask.shape

    def objective(theta: np.ndarray) -> float:
        params = BeakerParams.from_vector(theta.tolist())
        params = clip_beaker_params(params, image_shape[1], image_shape[0])

        rendered_mask = render_beaker_mask(params, image_shape, intrinsics)
        rendered_depth = render_beaker_depth(params, image_shape, intrinsics)

        l_mask = mask_iou_loss(rendered_mask, observed_mask)
        l_depth = robust_depth_loss(rendered_depth, observed_depth, observed_mask)
        l_contour = contour_chamfer_loss(rendered_mask, observed_mask)
        l_axis = axis_angle_loss(rendered_mask, observed_mask)
        l_top = top_width_loss(rendered_mask, observed_mask)
        l_prior = prior_regularization(params.radius, params.height)

        total = (
            4.0 * l_mask
            + 0.02 * l_depth
            + 0.4 * l_contour
            + 0.6 * l_axis
            + 0.6 * l_top
            + 0.1 * l_prior
        )
        return float(total)

    return objective


def overlay_masks(observed_mask: np.ndarray, rendered_mask: np.ndarray) -> np.ndarray:
    h, w = observed_mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[observed_mask > 0] = (0, 255, 0)
    overlap = (observed_mask > 0) & (rendered_mask > 0)
    canvas[rendered_mask > 0] = (255, 0, 0)
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

    observed_depth = load_depth(args.depth)
    if args.mask:
        observed_mask = load_mask(args.mask)
        observed_mask = (observed_mask > 0.5).astype(np.uint8)
    else:
        if args.interactive_roi:
            if not args.rgb:
                raise ValueError("--rgb is required when --interactive-roi is used")
            rgb = load_rgb(args.rgb)
            bbox = select_roi_bbox(rgb)
            observed_mask = make_mask_from_bbox(observed_depth.shape, bbox)
        elif args.bbox is not None:
            observed_mask = make_mask_from_bbox(observed_depth.shape, args.bbox)
        else:
            raise ValueError("Provide either --mask, or --bbox, or --interactive-roi with --rgb")

    intrinsics = CameraIntrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)

    init_params = initialize_beaker_from_mask_and_depth(observed_mask, observed_depth, intrinsics)
    objective = make_objective(observed_mask, observed_depth, intrinsics)

    result = minimize(
        objective,
        x0=np.asarray(init_params.to_vector(), dtype=np.float64),
        method="Powell",
        options={"maxiter": args.maxiter, "disp": True},
    )

    final_params = BeakerParams.from_vector(result.x.tolist())
    final_params = clip_beaker_params(final_params, observed_mask.shape[1], observed_mask.shape[0])
    rendered_mask = render_beaker_mask(final_params, observed_mask.shape, intrinsics)
    rendered_depth = render_beaker_depth(final_params, observed_mask.shape, intrinsics)

    summary = {
        "success": bool(result.success),
        "message": str(result.message),
        "final_loss": float(result.fun),
        "nfev": int(result.nfev),
        "initial_params": asdict(init_params),
        "final_params": asdict(final_params),
    }
    (Path(out_dir) / "fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cv2.imwrite(str(Path(out_dir) / "observed_mask.png"), (observed_mask * 255).astype(np.uint8))
    cv2.imwrite(str(Path(out_dir) / "rendered_mask.png"), (rendered_mask * 255).astype(np.uint8))
    cv2.imwrite(str(Path(out_dir) / "mask_overlay.png"), cv2.cvtColor(overlay_masks(observed_mask, rendered_mask), cv2.COLOR_RGB2BGR))
    save_depth_visual(Path(out_dir) / "observed_depth.png", observed_depth, observed_mask)
    save_depth_visual(Path(out_dir) / "rendered_depth.png", rendered_depth, rendered_mask)

    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
