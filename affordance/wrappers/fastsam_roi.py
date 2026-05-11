from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from fastsam import FastSAM, FastSAMPrompt

from affordance.utils.image_ops import bbox_from_mask


@dataclass
class FastSAMROIResult:
    roi_mask: np.ndarray
    roi_bbox: list[int]


class FastSAMROIGenerator:
    """Frozen FastSAM wrapper for ROI extraction from RGB images."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        imgsz: int = 1024,
        conf: float = 0.4,
        iou: float = 0.9,
        retina_masks: bool = True,
    ) -> None:
        checkpoint_path = self.resolve_checkpoint(checkpoint)
        self.model = FastSAM(str(checkpoint_path))
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.retina_masks = retina_masks

    @staticmethod
    def resolve_checkpoint(checkpoint: str) -> Path:
        requested = Path(checkpoint)
        if requested.exists():
            return requested

        candidates = []
        if requested.name == "FastSAM.pt":
            candidates.extend(
                [
                    requested.with_name("FastSAM-s.pt"),
                    Path("./weights/FastSAM-s.pt"),
                    Path("./FastSAM-s.pt"),
                ]
            )
        else:
            candidates.extend([Path("./weights/FastSAM-s.pt"), Path("./FastSAM-s.pt")])

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Unable to locate FastSAM checkpoint. Requested '{checkpoint}', "
            f"checked fallbacks: {[str(c) for c in candidates]}"
        )

    def run_everything(self, rgb_image: np.ndarray):
        image = Image.fromarray(rgb_image)
        return self.model(
            image,
            device=self.device,
            retina_masks=self.retina_masks,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
        )

    def mask_from_box(
        self,
        rgb_image: np.ndarray,
        bbox_xyxy: Iterable[int],
    ) -> FastSAMROIResult:
        bbox_xyxy = [int(v) for v in bbox_xyxy]
        results = self.run_everything(rgb_image)
        prompt = FastSAMPrompt(Image.fromarray(rgb_image), results, device=self.device)
        masks = prompt.box_prompt(bbox=bbox_xyxy)

        if len(masks) == 0:
            roi_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = bbox_xyxy
            roi_mask[y1:y2, x1:x2] = 1
            roi_bbox = bbox_xyxy
            return FastSAMROIResult(roi_mask=roi_mask, roi_bbox=roi_bbox)

        mask = np.array(masks[0]).astype(np.uint8)
        roi_bbox = bbox_from_mask(mask)
        return FastSAMROIResult(roi_mask=mask, roi_bbox=roi_bbox)
