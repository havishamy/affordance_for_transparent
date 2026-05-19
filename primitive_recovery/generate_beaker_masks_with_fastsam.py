from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from affordance.utils.image_ops import ensure_dir
from fastsam import FastSAM, FastSAMPrompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch segment the beaker in a RealSense sequence using FastSAM.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="e.g. /home/dsj/FastSAM/real_dataset/realsense_dataset/4")
    parser.add_argument("--rgb-subdir", type=str, default="rgb")
    parser.add_argument("--output-subdir", type=str, default="masks_fastsam_beaker")
    parser.add_argument("--preview-subdir", type=str, default="previews_fastsam_beaker")
    parser.add_argument("--checkpoint", type=str, default="./weights/FastSAM-s.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=0, help="0 means all images.")
    return parser.parse_args()


def center_box(width: int, height: int) -> list[int]:
    x1 = int(width * 0.20)
    y1 = int(height * 0.18)
    x2 = int(width * 0.82)
    y2 = int(height * 0.92)
    return [x1, y1, x2, y2]


def score_mask(mask: np.ndarray, target_xy: tuple[float, float], image_shape: tuple[int, int]) -> float:
    area = float(mask.sum())
    if area <= 0:
        return -1e9
    ys, xs = np.where(mask > 0)
    cx = float(xs.mean())
    cy = float(ys.mean())
    target_x, target_y = target_xy
    dist = ((cx - target_x) ** 2 + (cy - target_y) ** 2) ** 0.5
    h, w = image_shape
    area_ratio = area / float(h * w)
    # prefer a mid-size object near the center-lower part of the image
    return 3.0 * area_ratio - 0.005 * dist


def choose_mask_from_box_prompt(candidates: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray | None:
    if candidates is None:
        return None
    if isinstance(candidates, list):
        candidates = np.asarray(candidates)
    if candidates.ndim == 2:
        candidates = candidates[None, ...]
    if candidates.ndim != 3:
        return None

    h, w = image_shape
    target = (w * 0.5, h * 0.58)
    best_mask = None
    best_score = -1e18
    for i in range(candidates.shape[0]):
        mask = (candidates[i] > 0).astype(np.uint8)
        s = score_mask(mask, target, image_shape)
        if s > best_score:
            best_score = s
            best_mask = mask
    return best_mask


def choose_mask_from_everything(candidates, image_shape: tuple[int, int]) -> np.ndarray | None:
    if candidates is None:
        return None
    if hasattr(candidates, "cpu"):
        candidates = candidates.cpu().numpy()
    if candidates.ndim == 2:
        candidates = candidates[None, ...]
    if candidates.ndim != 3:
        return None

    h, w = image_shape
    target = (w * 0.5, h * 0.58)
    best_mask = None
    best_score = -1e18
    for i in range(candidates.shape[0]):
        mask = (candidates[i] > 0).astype(np.uint8)
        s = score_mask(mask, target, image_shape)
        if s > best_score:
            best_score = s
            best_mask = mask
    return best_mask


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = rgb.copy()
    overlay = np.zeros_like(color)
    overlay[mask > 0] = (0, 255, 0)
    out = cv2.addWeighted(color, 0.65, overlay, 0.35, 0)
    return out


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    rgb_dir = dataset_dir / args.rgb_subdir
    out_dir = ensure_dir(dataset_dir / args.output_subdir)
    preview_dir = ensure_dir(dataset_dir / args.preview_subdir)

    rgb_paths = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
    if args.limit > 0:
        rgb_paths = rgb_paths[: args.limit]
    if not rgb_paths:
        raise FileNotFoundError(f"No RGB images found in {rgb_dir}")

    model = FastSAM(args.checkpoint)

    for idx, rgb_path in enumerate(rgb_paths, start=1):
        image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[skip] unable to read {rgb_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        box = center_box(w, h)

        results = model(
            image_rgb,
            device=args.device,
            retina_masks=True,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
        )
        prompt = FastSAMPrompt(image_rgb, results, device=args.device)

        mask = None
        try:
            box_masks = prompt.box_prompt(bbox=box)
            mask = choose_mask_from_box_prompt(box_masks, (h, w))
        except Exception:
            mask = None

        if mask is None or mask.sum() == 0:
            try:
                everything = prompt.everything_prompt()
                mask = choose_mask_from_everything(everything, (h, w))
            except Exception:
                mask = None

        if mask is None or mask.sum() == 0:
            # last fallback: simple central bbox mask
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 1

        out_path = out_dir / f"{rgb_path.stem}.png"
        cv2.imwrite(str(out_path), (mask * 255).astype(np.uint8))

        prev = overlay_mask(image_rgb, mask)
        cv2.rectangle(prev, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)
        cv2.imwrite(str(preview_dir / f"{rgb_path.stem}.jpg"), cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))

        print(f"[{idx}/{len(rgb_paths)}] saved {out_path.name}")

    print(f"Done. Masks saved to {out_dir}")


if __name__ == "__main__":
    main()
