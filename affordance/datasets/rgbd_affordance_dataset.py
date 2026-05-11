from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import torch
from torch.utils.data import Dataset

from affordance.utils.image_ops import build_model_input, crop_array, load_depth, load_mask, load_rgb


class RGBDAffordanceDataset(Dataset):
    def __init__(
        self,
        annotation_file: str | Path,
        image_size: int = 256,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.image_size = image_size
        self.entries = []
        with self.annotation_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))
        if not self.entries:
            raise ValueError(f"No dataset entries found in {self.annotation_file}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]

        rgb = load_rgb(entry["rgb"])
        depth = load_depth(entry["depth"])
        roi_mask = load_mask(entry["roi_mask"])
        affordance_mask = load_mask(entry["affordance_mask"])
        bbox = entry["roi_bbox"]

        rgb_crop = crop_array(rgb, bbox)
        depth_crop = crop_array(depth, bbox)
        roi_crop = crop_array(roi_mask, bbox)
        affordance_crop = crop_array(affordance_mask, bbox)

        image_tensor = build_model_input(
            rgb_crop=rgb_crop,
            depth_crop=depth_crop,
            roi_mask_crop=roi_crop,
            image_size=(self.image_size, self.image_size),
        )
        affordance_resized = cv2.resize(
            affordance_crop.astype("float32"),
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        target = torch.from_numpy(affordance_resized).unsqueeze(0).float()

        return {
            "image": image_tensor.float(),
            "target": target,
            "instruction": entry["instruction"],
            "object_name": entry.get("object_name", ""),
            "entry": entry,
        }

