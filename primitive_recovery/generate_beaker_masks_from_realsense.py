from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from affordance.utils.image_ops import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate coarse beaker masks from a RealSense sequence with a single beaker.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path like /home/dsj/FastSAM/real_dataset/realsense_dataset/4")
    parser.add_argument("--depth-subdir", type=str, default="depth_completed", help="Use completed depth by default.")
    parser.add_argument("--mask-subdir", type=str, default="masks_auto")
    parser.add_argument("--depth-thresh-mm", type=int, default=0, help="Optional fixed threshold in mm. 0 means adaptive threshold.")
    parser.add_argument("--quantile", type=float, default=12.0, help="Adaptive foreground threshold quantile for nearest pixels.")
    parser.add_argument("--pad-mm", type=float, default=10.0, help="Extra margin added to adaptive threshold.")
    parser.add_argument("--open-kernel", type=int, default=5)
    parser.add_argument("--close-kernel", type=int, default=9)
    return parser.parse_args()


def largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    out = np.zeros_like(mask, dtype=np.uint8)
    out[labels == idx] = 1
    return out


def estimate_threshold(depth: np.ndarray, quantile: float, pad_mm: float) -> int:
    vals = depth[depth > 0]
    if vals.size == 0:
        return 0
    base = float(np.percentile(vals, quantile))
    return int(base + pad_mm)


def refine_mask(depth: np.ndarray, threshold_mm: int, quantile: float, pad_mm: float, open_kernel: int, close_kernel: int) -> np.ndarray:
    if threshold_mm <= 0:
        threshold_mm = estimate_threshold(depth, quantile, pad_mm)
    mask = ((depth > 0) & (depth < threshold_mm)).astype(np.uint8)
    mask = largest_component(mask)

    open_k = np.ones((open_kernel, open_kernel), np.uint8)
    close_k = np.ones((close_kernel, close_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    mask = largest_component(mask)
    return mask


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    depth_dir = dataset_dir / args.depth_subdir
    out_dir = ensure_dir(dataset_dir / args.mask_subdir)

    depth_files = sorted(depth_dir.glob("*.png"))
    if not depth_files:
        raise FileNotFoundError(f"No depth images found in {depth_dir}")

    for depth_path in depth_files:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        if depth.ndim == 3:
            depth = depth[..., 0]
        mask = refine_mask(
            depth=depth.astype(np.uint16),
            threshold_mm=args.depth_thresh_mm,
            quantile=args.quantile,
            pad_mm=args.pad_mm,
            open_kernel=args.open_kernel,
            close_kernel=args.close_kernel,
        )
        out_path = out_dir / depth_path.name
        cv2.imwrite(str(out_path), (mask * 255).astype(np.uint8))

    print(f"Saved masks to {out_dir}")


if __name__ == "__main__":
    main()
