from __future__ import annotations

import cv2
import numpy as np

from primitive_recovery.geometry import estimate_top_width, principal_axis


def mask_iou_loss(rendered_mask: np.ndarray, observed_mask: np.ndarray, eps: float = 1e-6) -> float:
    rendered = rendered_mask > 0
    observed = observed_mask > 0
    inter = np.logical_and(rendered, observed).sum()
    union = np.logical_or(rendered, observed).sum()
    return 1.0 - float(inter + eps) / float(union + eps)


def robust_depth_loss(
    rendered_depth: np.ndarray,
    observed_depth: np.ndarray,
    observed_mask: np.ndarray,
    delta: float = 20.0,
) -> float:
    valid = (observed_mask > 0) & (observed_depth > 0) & (rendered_depth > 0)
    if not np.any(valid):
        return 0.0
    diff = rendered_depth[valid] - observed_depth[valid]
    abs_diff = np.abs(diff)
    quadratic = np.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    huber = 0.5 * quadratic**2 + delta * linear
    return float(np.mean(huber))


def contour_chamfer_loss(rendered_mask: np.ndarray, observed_mask: np.ndarray) -> float:
    rendered_u8 = (rendered_mask > 0).astype(np.uint8) * 255
    observed_u8 = (observed_mask > 0).astype(np.uint8) * 255
    rendered_edges = cv2.Canny(rendered_u8, 50, 150)
    observed_edges = cv2.Canny(observed_u8, 50, 150)

    if rendered_edges.sum() == 0 or observed_edges.sum() == 0:
        return 1e3

    dist_to_obs = cv2.distanceTransform(255 - observed_edges, cv2.DIST_L2, 3)
    dist_to_rnd = cv2.distanceTransform(255 - rendered_edges, cv2.DIST_L2, 3)
    rnd_pts = rendered_edges > 0
    obs_pts = observed_edges > 0
    loss = float(dist_to_obs[rnd_pts].mean() + dist_to_rnd[obs_pts].mean()) * 0.5
    return loss


def axis_angle_loss(rendered_mask: np.ndarray, observed_mask: np.ndarray) -> float:
    _, axis_r = principal_axis(rendered_mask)
    _, axis_o = principal_axis(observed_mask)
    axis_r = axis_r / (np.linalg.norm(axis_r) + 1e-8)
    axis_o = axis_o / (np.linalg.norm(axis_o) + 1e-8)
    cosine = float(np.clip(np.abs(np.dot(axis_r, axis_o)), 0.0, 1.0))
    return 1.0 - cosine


def top_width_loss(rendered_mask: np.ndarray, observed_mask: np.ndarray) -> float:
    wr = estimate_top_width(rendered_mask)
    wo = estimate_top_width(observed_mask)
    if wo <= 1e-6:
        return 0.0
    return abs(wr - wo) / wo


def prior_regularization(radius: float, height: float) -> float:
    penalties = 0.0
    if radius <= 0:
        penalties += 1000.0
    if height <= 0:
        penalties += 1000.0
    ratio = height / max(radius, 1e-6)
    if ratio < 0.8:
        penalties += (0.8 - ratio) * 50.0
    if ratio > 8.0:
        penalties += (ratio - 8.0) * 10.0
    return float(penalties)

