from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from affordance.infer_chem_hova import infer_one, load_checkpoint
from affordance.models.text_encoder import MiniLMTextEncoder
from affordance.utils.image_ops import ensure_dir, load_rgb
from affordance.utils.visualize import overlay_heatmap_on_rgb, save_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference over the full chem_hova test set.")
    parser.add_argument("--dataset-root", type=str, default="/home/dsj/FastSAM/chem_hova_dataset")
    parser.add_argument(
        "--annotation-file",
        type=str,
        default=None,
        help="Optional explicit test annotation json path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/dsj/FastSAM/affordance_runs/chem_hova_fullimg_e5/best.pt",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/home/dsj/FastSAM/affordance_outputs/chem_hova_test_all",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of samples. 0 means all.")
    return parser.parse_args()


def build_instruction(noun: str, action: str) -> str:
    return f"Where should I interact with the {noun} to {action.replace('_', ' ')} it?"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset_root = Path(args.dataset_root).resolve()
    annotation_file = (
        Path(args.annotation_file).resolve()
        if args.annotation_file is not None
        else dataset_root / "annotations" / "test" / "chem_lab_test.json"
    )
    checkpoint = Path(args.checkpoint).resolve()
    output_root = ensure_dir(args.output_root)

    with annotation_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if args.limit > 0:
        data = data[: args.limit]

    model, text_model_name, image_size = load_checkpoint(str(checkpoint), device)
    text_encoder = MiniLMTextEncoder(text_model_name, freeze=True).to(device)

    total = len(data)
    for idx, item in enumerate(data, start=1):
        rgb_path = dataset_root / item["img_path"]
        rgb = load_rgb(rgb_path)
        instruction = build_instruction(item["noun"], item["action"])
        sample_name = f"{idx:05d}_{item['object']}_{item['action']}"
        out_dir = ensure_dir(output_root / sample_name)

        heatmap, (point_x, point_y) = infer_one(
            model=model,
            text_encoder=text_encoder,
            rgb=rgb,
            instruction=instruction,
            image_size=image_size,
            device=device,
        )

        overlay = overlay_heatmap_on_rgb(rgb, heatmap)
        overlay = overlay.copy()
        cv2.circle(overlay, (point_x, point_y), 6, (255, 255, 255), -1)

        np.save(out_dir / "affordance_heatmap.npy", heatmap)
        cv2.imwrite(str(out_dir / "affordance_heatmap.png"), np.clip(heatmap * 255.0, 0, 255).astype(np.uint8))
        save_rgb(out_dir / "affordance_overlay.png", overlay)

        print(f"[{idx}/{total}] saved {out_dir.name}")


if __name__ == "__main__":
    main()
