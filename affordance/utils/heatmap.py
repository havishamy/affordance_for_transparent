from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch


def make_gaussian_heatmap(
    height: int,
    width: int,
    points: Sequence[Sequence[float]],
    sigma: float,
    roi_mask: np.ndarray | None = None,
) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    yy, xx = np.mgrid[0:height, 0:width]
    heatmap = np.zeros((height, width), dtype=np.float32)
    for point in points:
        x, y = float(point[0]), float(point[1])
        point_map = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))
        heatmap = np.maximum(heatmap, point_map.astype(np.float32))
    if roi_mask is not None:
        heatmap = heatmap * (roi_mask > 0).astype(np.float32)
    max_value = float(heatmap.max())
    if max_value > 0:
        heatmap /= max_value
    return heatmap


def sigmoid_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    numerator = 2.0 * (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - ((numerator + eps) / (denominator + eps)).mean()


def top_point_from_heatmap(heatmap: np.ndarray, roi_mask: np.ndarray | None = None) -> tuple[int, int]:
    if roi_mask is not None:
        masked = heatmap * (roi_mask > 0)
    else:
        masked = heatmap
    flat_index = int(np.argmax(masked))
    y, x = np.unravel_index(flat_index, masked.shape)
    return int(x), int(y)

