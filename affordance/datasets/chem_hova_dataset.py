from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import torch
from torch.utils.data import Dataset

from affordance.utils.image_ops import load_mask, load_rgb


def _format_instruction(noun: str, action: str) -> str:
    action_text = action.replace("_", " ")
    return f"Where should I interact with the {noun} to {action_text} it?"


class ChemHOVADataset(Dataset):
    """Full-image dataset for chem_hova affordance heatmap learning."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        image_size: int = 384,
        annotation_file: str | Path | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        self.annotation_file = (
            Path(annotation_file)
            if annotation_file is not None
            else self.dataset_root / "annotations" / split / f"chem_lab_{split}.json"
        )
        with self.annotation_file.open("r", encoding="utf-8") as handle:
            self.entries = json.load(handle)
        if not isinstance(self.entries, list) or not self.entries:
            raise ValueError(f"No dataset entries found in {self.annotation_file}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        rgb_path = self.dataset_root / entry["img_path"]
        gt_path = self.dataset_root / "GT_gaussian" / self.split / entry["gt_path"]

        rgb = load_rgb(rgb_path)
        target = load_mask(gt_path)

        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(target).unsqueeze(0).float()

        return {
            "image": rgb_tensor,
            "target": target_tensor,
            "instruction": _format_instruction(entry["noun"], entry["action"]),
            "object_name": entry.get("object", entry.get("noun", "")),
            "bbox": entry.get("bbox"),
            "points": entry.get("points", []),
            "entry": entry,
        }
