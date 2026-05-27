"""
inference/centroid.py — Centroid extraction and live sensor frame utilities.

Provides soft/hard argmax centroid extraction from model output tensors,
and the mouse-to-sensor-frame pipeline used by the live demo.
"""
from __future__ import annotations

import os
import sys
from typing import Tuple, Union

import numpy as np
import torch

# Allow imports from project root regardless of CWD
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.noise import apply_all_noise  # noqa: E402
from data.generate_dataset import make_gaussian_heatmap  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
GRID_SIZE: int = 20
BLOB_SIGMA: float = 1.5


def soft_argmax(heatmap: np.ndarray) -> Tuple[float, float]:
    """Compute the soft-argmax (weighted centroid) of a 2-D heatmap.

    Parameters
    ----------
    heatmap : np.ndarray, shape (H, W), float32

    Returns
    -------
    (cx, cy) — column and row centroid as floats.
    Normalizes by the peak value first so low-amplitude model outputs
    (near zero) still produce a meaningful centroid rather than falling
    back to the image center.
    Falls back to hard-argmax peak if the heatmap is essentially zero.
    """
    h, w = heatmap.shape

    # Normalize by peak so amplitude doesn't affect the centroid
    peak = float(heatmap.max())
    if peak < 1e-8:
        # Truly empty — use hard-argmax peak position
        flat_idx = int(np.argmax(heatmap))
        row, col = np.unravel_index(flat_idx, heatmap.shape)
        return float(col), float(row)

    normed = heatmap / peak  # values in [0, 1]
    total = float(normed.sum())

    col_indices = np.arange(w, dtype=np.float32)[np.newaxis, :]  # (1, W)
    row_indices = np.arange(h, dtype=np.float32)[:, np.newaxis]  # (H, 1)

    cx = float((normed * col_indices).sum() / total)
    cy = float((normed * row_indices).sum() / total)
    return cx, cy


def hard_argmax(heatmap: np.ndarray) -> Tuple[int, int]:
    """Return the integer (cx, cy) coordinates of the heatmap peak.

    Parameters
    ----------
    heatmap : np.ndarray, shape (H, W)

    Returns
    -------
    (cx, cy) as ints (column, row).
    """
    flat_idx = int(np.argmax(heatmap))
    row, col = np.unravel_index(flat_idx, heatmap.shape)
    return int(col), int(row)


def extract_centroid_from_tensor(
    pred_tensor: torch.Tensor,
    method: str = "soft",
) -> Tuple[float, float]:
    """Extract (cx, cy) centroid from a model output tensor.

    Accepts tensors of shape (1,1,H,W), (1,H,W), or (H,W).
    """
    heatmap = pred_tensor.detach().cpu().squeeze().numpy().astype(np.float32)
    if method == "soft":
        return soft_argmax(heatmap)
    return tuple(float(v) for v in hard_argmax(heatmap))  # type: ignore[return-value]


def mouse_to_sensor_frame(
    mouse_x: int,
    mouse_y: int,
    screen_w: int,
    screen_h: int,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """Map mouse screen coordinates to a noisy (1, grid_size, grid_size) sensor frame.

    1. Converts pixel position to grid coords in [0.5, grid_size - 1.5].
    2. Generates a clean Gaussian blob at that location.
    3. Applies full sensor noise pipeline (apply_all_noise).
    4. Returns float32 array of shape (1, grid_size, grid_size).
    """
    # Map to continuous grid coords
    cx = (mouse_x / screen_w) * grid_size
    cy = (mouse_y / screen_h) * grid_size

    # Clamp away from borders so the blob is always fully visible
    cx = float(np.clip(cx, 0.5, grid_size - 1.5))
    cy = float(np.clip(cy, 0.5, grid_size - 1.5))

    clean = make_gaussian_heatmap(cx, cy, size=grid_size, sigma=BLOB_SIGMA)
    noisy = apply_all_noise(clean)
    return noisy[np.newaxis, :, :].astype(np.float32)  # (1, H, W)


def frame_to_tensor(
    frame: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Convert a (1, H, W) numpy frame to a (1, 1, H, W) model-ready tensor.

    Parameters
    ----------
    frame  : np.ndarray, shape (1, H, W), float32
    device : torch.device

    Returns
    -------
    torch.Tensor, shape (1, 1, H, W)
    """
    tensor = torch.from_numpy(frame).unsqueeze(0).to(device)  # (1, 1, H, W)
    return tensor
