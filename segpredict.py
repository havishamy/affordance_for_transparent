from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from fastsam import FastSAM, FastSAMPrompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch segment the beaker in a RealSense RGB folder with FastSAM.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/dsj/FastSAM/weights/FastSAM-s.pt",
        help="Path to FastSAM checkpoint.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/home/dsj/FastSAM/real_dataset/realsense_dataset/4/rgb",
        help="Directory containing RGB images.",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="/home/dsj/FastSAM/real_dataset/realsense_dataset/4/masks_fastsam_beaker",
        help="Output directory for binary masks.",
    )
    parser.add_argument(
        "--preview-dir",
        type=str,
        default="/home/dsj/FastSAM/real_dataset/realsense_dataset/4/previews_fastsam_beaker",
        help="Output directory for RGB overlays.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of images. 0 means all.")
    return parser.parse_args()


def center_box(width: int, height: int) -> list[int]:
    # The beaker is placed roughly in the image center and lower half in this sequence.
    x1 = int(width * 0.22)
    y1 = int(height * 0.18)
    x2 = int(width * 0.78)
    y2 = int(height * 0.95)
    return [x1, y1, x2, y2]


def score_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> float:
    area = float(mask.sum())
    if area <= 0:
        return -1e9
    h, w = image_shape
    ys, xs = np.where(mask > 0)
    cx = float(xs.mean())
    cy = float(ys.mean())
    target_x = w * 0.5
    target_y = h * 0.60
    center_dist = ((cx - target_x) ** 2 + (cy - target_y) ** 2) ** 0.5
    area_ratio = area / float(h * w)
    # Prefer a medium-size object near the center-lower region.
    return 4.0 * area_ratio - 0.005 * center_dist


def choose_best_mask(candidates, image_shape: tuple[int, int]) -> np.ndarray | None:
    if candidates is None:
        return None
    if hasattr(candidates, "cpu"):
        candidates = candidates.cpu().numpy()
    candidates = np.asarray(candidates)
    if candidates.ndim == 2:
        candidates = candidates[None, ...]
    if candidates.ndim != 3:
        return None

    best = None
    best_score = -1e18
    for i in range(candidates.shape[0]):
        mask = (candidates[i] > 0).astype(np.uint8)
        s = score_mask(mask, image_shape)
        if s > best_score:
            best_score = s
            best = mask
    return best


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, box: list[int]) -> np.ndarray:
    vis = rgb.copy()
    tint = np.zeros_like(vis)
    tint[mask > 0] = (0, 255, 0)
    vis = cv2.addWeighted(vis, 0.65, tint, 0.35, 0)
    cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)
    return vis


def main() -> None:
    args = parse_args()

    model = FastSAM(args.model_path)
    input_dir = Path(args.input_dir)
    mask_dir = Path(args.mask_dir)
    preview_dir = Path(args.preview_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    image_paths = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    for idx, image_path in enumerate(image_paths, start=1):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[skip] unable to read {image_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        box = center_box(w, h)

        everything_results = model(
            image_rgb,
            device=device,
            retina_masks=True,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
        )
        prompt_process = FastSAMPrompt(image_rgb, everything_results, device=device)

        mask = None
        try:
            ann = prompt_process.box_prompt(bbox=box)
            mask = choose_best_mask(ann, (h, w))
        except Exception:
            mask = None

        if mask is None or mask.sum() == 0:
            try:
                ann = prompt_process.everything_prompt()
                mask = choose_best_mask(ann, (h, w))
            except Exception:
                mask = None

        if mask is None or mask.sum() == 0:
            # Last fallback: use the ROI box as a coarse mask.
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 1

        mask_path = mask_dir / f"{image_path.stem}.png"
        preview_path = preview_dir / f"{image_path.stem}.jpg"

        cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
        preview = overlay_mask(image_rgb, mask, box)
        cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

        print(f"[{idx}/{len(image_paths)}] saved {mask_path.name}")

    print(f"Done. Masks saved to {mask_dir}")


if __name__ == "__main__":
    main()
