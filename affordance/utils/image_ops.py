from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_depth(path: str | Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Unable to read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def load_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read mask image: {path}")
    return mask.astype(np.float32) / 255.0


def normalize_depth(depth: np.ndarray, invalid_fill: float = 0.0) -> np.ndarray:
    depth = depth.astype(np.float32)
    valid = depth > 0
    if not np.any(valid):
        return np.full_like(depth, invalid_fill, dtype=np.float32)
    valid_values = depth[valid]
    min_value = float(valid_values.min())
    max_value = float(valid_values.max())
    if max_value - min_value < 1e-6:
        normalized = np.zeros_like(depth, dtype=np.float32)
        normalized[valid] = 1.0
        return normalized
    normalized = np.zeros_like(depth, dtype=np.float32)
    normalized[valid] = (depth[valid] - min_value) / (max_value - min_value)
    normalized[~valid] = invalid_fill
    return normalized


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("mask is empty")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def crop_array(array: np.ndarray, bbox: Iterable[int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return array[y1:y2, x1:x2]


def resize_array(array: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
    width, height = size
    return cv2.resize(array, (width, height), interpolation=interpolation)


def build_model_input(
    rgb_crop: np.ndarray,
    depth_crop: np.ndarray,
    roi_mask_crop: np.ndarray,
    image_size: tuple[int, int],
) -> torch.Tensor:
    width, height = image_size
    rgb_resized = resize_array(rgb_crop, image_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    depth_resized = resize_array(depth_crop, image_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    depth_resized = normalize_depth(depth_resized)
    roi_resized = resize_array(roi_mask_crop, image_size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    if roi_resized.max() > 1.0:
        roi_resized /= 255.0

    rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1)
    depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0)
    roi_tensor = torch.from_numpy(roi_resized).unsqueeze(0)
    return torch.cat([rgb_tensor, depth_tensor, roi_tensor], dim=0)

