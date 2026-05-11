from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from affordance.models.glover_lite_fullimg import GloverLiteFullImageNet
from affordance.models.text_encoder import MiniLMTextEncoder
from affordance.utils.heatmap import top_point_from_heatmap
from affordance.utils.image_ops import ensure_dir, load_rgb
from affordance.utils.visualize import overlay_heatmap_on_rgb, save_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-image chem_hova affordance inference.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--rgb", type=str, required=True)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="affordance_outputs/chem_hova")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_checkpoint(path: str, device: torch.device) -> tuple[GloverLiteFullImageNet, str, int]:
    checkpoint = torch.load(path, map_location=device)
    model_kwargs = checkpoint.get("model_kwargs", {"text_dim": 384, "backbone_pretrained": False})
    model = GloverLiteFullImageNet(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    text_model_name = checkpoint.get("text_model_name", "sentence-transformers/all-MiniLM-L6-v2")
    image_size = int(checkpoint.get("image_size", 384))
    return model, text_model_name, image_size


def infer_one(
    model: GloverLiteFullImageNet,
    text_encoder: MiniLMTextEncoder,
    rgb: np.ndarray,
    instruction: str,
    image_size: int,
    device: torch.device,
) -> tuple[np.ndarray, tuple[int, int]]:
    original_h, original_w = rgb.shape[:2]
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        text_features = text_encoder([instruction], device=device)
        logits = model(image_tensor, text_features)
        heatmap = torch.sigmoid(logits)[0, 0].cpu().numpy()

    heatmap = cv2.resize(heatmap, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    point_x, point_y = top_point_from_heatmap(heatmap)
    return heatmap, (point_x, point_y)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = ensure_dir(args.output_dir)

    rgb = load_rgb(args.rgb)
    model, text_model_name, image_size = load_checkpoint(args.checkpoint, device)
    text_encoder = MiniLMTextEncoder(text_model_name, freeze=True).to(device)
    heatmap, (point_x, point_y) = infer_one(
        model=model,
        text_encoder=text_encoder,
        rgb=rgb,
        instruction=args.instruction,
        image_size=image_size,
        device=device,
    )

    overlay = overlay_heatmap_on_rgb(rgb, heatmap)
    overlay = overlay.copy()
    cv2.circle(overlay, (point_x, point_y), 6, (255, 255, 255), -1)

    np.save(out_dir / "affordance_heatmap.npy", heatmap)
    cv2.imwrite(str(out_dir / "affordance_heatmap.png"), np.clip(heatmap * 255.0, 0, 255).astype(np.uint8))
    save_rgb(out_dir / "affordance_overlay.png", overlay)

    print(f"Saved outputs to {out_dir}")
    print(f"Predicted top affordance point: ({point_x}, {point_y})")


if __name__ == "__main__":
    main()
