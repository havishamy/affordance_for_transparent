from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .image_ops import ensure_dir


def overlay_heatmap_on_rgb(
    rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    heatmap_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(rgb, 1.0 - alpha, color_map, alpha, 0)
    return blended


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    ensure_dir(Path(path).parent)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)

