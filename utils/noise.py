"""
utils/noise.py — Sensor noise simulation for touch-heatmap-reconstruction.

All noise types applied during dataset generation and live demo.
Standalone module — no project imports.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def add_gaussian_noise(
    frame: np.ndarray,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """Add zero-mean Gaussian noise with randomised or fixed sigma."""
    if sigma is None:
        sigma = float(np.random.uniform(0.05, 0.15))
    noise = np.random.normal(0.0, sigma, frame.shape).astype(np.float32)
    return (frame + noise).astype(np.float32)


def add_salt_and_pepper(
    frame: np.ndarray,
    ratio: Optional[float] = None,
) -> np.ndarray:
    """Add salt-and-pepper impulse noise with randomised or fixed ratio."""
    if ratio is None:
        ratio = float(np.random.uniform(0.01, 0.05))

    result = frame.copy().astype(np.float32)
    total_pixels = frame.size
    n_affected = int(total_pixels * ratio)

    # --- Salt (set to 1.0) ---
    flat_indices_salt = np.random.choice(total_pixels, n_affected // 2, replace=False)
    result_flat = result.ravel()
    result_flat[flat_indices_salt] = 1.0

    # --- Pepper (set to 0.0) ---
    remaining = np.setdiff1d(np.arange(total_pixels), flat_indices_salt)
    flat_indices_pepper = np.random.choice(remaining, n_affected // 2, replace=False)
    result_flat[flat_indices_pepper] = 0.0

    return result.astype(np.float32)


def add_baseline_drift(
    frame: np.ndarray,
    amplitude: Optional[float] = None,
) -> np.ndarray:
    """Add a low-frequency sinusoidal 2-D drift surface to the frame."""
    if amplitude is None:
        amplitude = float(np.random.uniform(0.0, 0.10))

    rows, cols = frame.shape
    row_freq = float(np.random.uniform(0.5, 2.0))
    col_freq = float(np.random.uniform(0.5, 2.0))

    row_wave = np.sin(
        2.0 * np.pi * row_freq * np.arange(rows, dtype=np.float32) / rows
    )
    col_wave = np.sin(
        2.0 * np.pi * col_freq * np.arange(cols, dtype=np.float32) / cols
    )
    drift = amplitude * np.outer(row_wave, col_wave).astype(np.float32)
    return (frame + drift).astype(np.float32)


def add_quantization_noise(
    frame: np.ndarray,
    bits: int = 8,
) -> np.ndarray:
    """Quantise the frame to a fixed number of discrete levels."""
    levels = float(2 ** bits - 1)
    quantised = (np.round(frame * levels) / levels).astype(np.float32)
    return quantised


def apply_all_noise(frame: np.ndarray) -> np.ndarray:
    """Apply all noise types in order: drift → gaussian → s&p → quantization.

    Clips the final output to [0.0, 1.0].  This is the single entry point
    used everywhere else in the project.
    """
    result = frame.astype(np.float32)
    result = add_baseline_drift(result)
    result = add_gaussian_noise(result)
    result = add_salt_and_pepper(result)
    result = add_quantization_noise(result)
    result = np.clip(result, 0.0, 1.0).astype(np.float32)
    return result
