from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from affordance.models.roi_affordance_net import ROIAffordanceNet
from affordance.models.text_encoder import MiniLMTextEncoder
from affordance.utils.heatmap import top_point_from_heatmap
from affordance.utils.image_ops import (
    bbox_from_mask,
    build_model_input,
    crop_array,
    ensure_dir,
    load_depth,
    load_mask,
    load_rgb,
)
from affordance.utils.visualize import overlay_heatmap_on_rgb, save_rgb
from affordance.wrappers.fastsam_roi import FastSAMROIGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone RGB-D affordance inference.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--rgb", type=str, required=True)
    parser.add_argument("--depth", type=str, required=True)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="affordance_outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fastsam-checkpoint", type=str, default="./weights/FastSAM.pt")
    parser.add_argument("--fastsam-imgsz", type=int, default=1024)
    parser.add_argument("--interactive-roi", action="store_true", default=False)
    parser.add_argument("--roi-bbox", type=int, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--roi-mask", type=str, default=None, help="Optional precomputed ROI mask path.")
    return parser.parse_args()


def select_roi_bbox(rgb: np.ndarray) -> list[int]:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    x, y, w, h = cv2.selectROI("Select ROI", bgr, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    if w <= 0 or h <= 0:
        raise ValueError("ROI selection was cancelled")
    return [int(x), int(y), int(x + w), int(y + h)]


def load_checkpoint(path: str, device: torch.device) -> tuple[ROIAffordanceNet, str, int]:
    checkpoint = torch.load(path, map_location=device)
    model_kwargs = checkpoint.get("model_kwargs", {"text_dim": 384})
    model = ROIAffordanceNet(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    text_model_name = checkpoint.get("text_model_name", "sentence-transformers/all-MiniLM-L6-v2")
    image_size = int(checkpoint.get("image_size", 256))
    return model, text_model_name, image_size


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = ensure_dir(args.output_dir)

    rgb = load_rgb(args.rgb)
    depth = load_depth(args.depth)
    model, text_model_name, image_size = load_checkpoint(args.checkpoint, device)
    text_encoder = MiniLMTextEncoder(text_model_name, freeze=True).to(device)

    if args.roi_mask:
        roi_mask = load_mask(args.roi_mask)
        roi_mask = (roi_mask > 0.5).astype(np.uint8)
        roi_bbox = bbox_from_mask(roi_mask)
    else:
        if args.roi_bbox is not None:
            selected_bbox = args.roi_bbox
        elif args.interactive_roi:
            selected_bbox = select_roi_bbox(rgb)
        else:
            raise ValueError("Provide --roi-mask, --roi-bbox, or --interactive-roi")

        fastsam = FastSAMROIGenerator(
            checkpoint=args.fastsam_checkpoint,
            device=args.device,
            imgsz=args.fastsam_imgsz,
        )
        roi_result = fastsam.mask_from_box(rgb, selected_bbox)
        roi_mask = roi_result.roi_mask.astype(np.uint8)
        roi_bbox = roi_result.roi_bbox

    rgb_crop = crop_array(rgb, roi_bbox)
    depth_crop = crop_array(depth, roi_bbox)
    roi_crop = crop_array(roi_mask.astype(np.float32), roi_bbox)

    image_tensor = build_model_input(
        rgb_crop=rgb_crop,
        depth_crop=depth_crop,
        roi_mask_crop=roi_crop,
        image_size=(image_size, image_size),
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        text_features = text_encoder([args.instruction], device=device)
        logits = model(image_tensor, text_features)
        heatmap_crop = torch.sigmoid(logits)[0, 0].cpu().numpy()

    crop_h = roi_bbox[3] - roi_bbox[1]
    crop_w = roi_bbox[2] - roi_bbox[0]
    heatmap_crop = cv2.resize(heatmap_crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    heatmap_crop = heatmap_crop * roi_crop

    full_heatmap = np.zeros(rgb.shape[:2], dtype=np.float32)
    full_heatmap[roi_bbox[1] : roi_bbox[3], roi_bbox[0] : roi_bbox[2]] = heatmap_crop
    point_x, point_y = top_point_from_heatmap(full_heatmap, roi_mask=roi_mask)

    overlay = overlay_heatmap_on_rgb(rgb, full_heatmap)
    overlay = overlay.copy()
    cv2.circle(overlay, (point_x, point_y), 6, (255, 255, 255), -1)

    np.save(out_dir / "affordance_heatmap.npy", full_heatmap)
    cv2.imwrite(str(out_dir / "affordance_heatmap.png"), (full_heatmap * 255.0).astype(np.uint8))
    cv2.imwrite(str(out_dir / "roi_mask.png"), (roi_mask * 255).astype(np.uint8))
    save_rgb(out_dir / "affordance_overlay.png", overlay)

    print(f"Saved outputs to {out_dir}")
    print(f"Predicted top affordance point: ({point_x}, {point_y})")


if __name__ == "__main__":
    main()
