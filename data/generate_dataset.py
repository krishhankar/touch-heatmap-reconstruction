"""
data/generate_dataset.py — Synthetic dataset generator.

Produces 50 000 (noisy_frame, clean_heatmap) pairs and saves them to
data/dataset.npz as train/val splits ready for model/train.py.

Usage:
    python data/generate_dataset.py
    python data/generate_dataset.py --samples 100000 --output data/big.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
from tqdm import tqdm

# Allow imports from the project root regardless of CWD
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.noise import apply_all_noise  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
GRID_SIZE: int = 20
BLOB_SIGMA: float = 1.5


def make_gaussian_heatmap(
    cx: float,
    cy: float,
    size: int = GRID_SIZE,
    sigma: float = BLOB_SIGMA,
) -> np.ndarray:
    """Return a (size, size) float32 Gaussian blob centred at (cx, cy).

    Peak value is 1.0 at the centre; values decay to ~0 at the edges.
    """
    col_grid, row_grid = np.meshgrid(
        np.arange(size, dtype=np.float32),
        np.arange(size, dtype=np.float32),
    )
    heatmap = np.exp(
        -((col_grid - cx) ** 2 + (row_grid - cy) ** 2) / (2.0 * sigma ** 2)
    ).astype(np.float32)
    return heatmap


def generate_sample(
    grid_size: int = GRID_SIZE,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Generate a single (noisy, clean, cx, cy) training sample.

    Returns
    -------
    noisy : np.ndarray, shape (grid_size, grid_size), float32
    clean : np.ndarray, shape (grid_size, grid_size), float32
    cx    : float — column (x) centroid in grid coords
    cy    : float — row    (y) centroid in grid coords
    """
    cx = float(np.random.uniform(1.0, grid_size - 2.0))
    cy = float(np.random.uniform(1.0, grid_size - 2.0))
    clean = make_gaussian_heatmap(cx, cy, size=grid_size)
    noisy = apply_all_noise(clean)
    return noisy, clean, cx, cy


def generate_dataset(
    n_samples: int = 50_000,
    grid_size: int = GRID_SIZE,
    val_split: float = 0.1,
    output_path: str = "data/dataset.npz",
    seed: int = 42,
) -> None:
    """Generate the full dataset and save it as a compressed .npz archive.

    Keys in the archive:
        train_noisy, train_clean, train_pos  (shape: N_train × 1 × H × W)
        val_noisy,   val_clean,   val_pos    (shape: N_val   × 1 × H × W)
    """
    np.random.seed(seed)

    # Pre-allocate arrays with channel dimension
    noisy_data = np.zeros((n_samples, 1, grid_size, grid_size), dtype=np.float32)
    clean_data = np.zeros((n_samples, 1, grid_size, grid_size), dtype=np.float32)
    positions = np.zeros((n_samples, 2), dtype=np.float32)

    print(f"Generating {n_samples:,} samples …")
    for i in tqdm(range(n_samples), unit="sample", ncols=80):
        noisy, clean, cx, cy = generate_sample(grid_size)
        noisy_data[i, 0] = noisy   # insert into channel dim
        clean_data[i, 0] = clean
        positions[i, 0] = cx
        positions[i, 1] = cy

    # Train / val split
    n_train = int(n_samples * (1.0 - val_split))
    train_noisy = noisy_data[:n_train]
    train_clean = clean_data[:n_train]
    train_pos = positions[:n_train]
    val_noisy = noisy_data[n_train:]
    val_clean = clean_data[n_train:]
    val_pos = positions[n_train:]

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving to {output_path} …")
    np.savez_compressed(
        output_path,
        train_noisy=train_noisy,
        train_clean=train_clean,
        train_pos=train_pos,
        val_noisy=val_noisy,
        val_clean=val_clean,
        val_pos=val_pos,
    )

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"Done. File size: {size_mb:.1f} MB")
    print(f"  Train: {n_train:,} samples | Val: {n_samples - n_train:,} samples")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic touch-sensor dataset."
    )
    parser.add_argument(
        "--samples", type=int, default=50_000,
        help="Total number of samples (default: 50000)",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("data", "dataset.npz"),
        help="Output .npz path (default: data/dataset.npz)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1,
        help="Fraction reserved for validation (default: 0.1)",
    )
    args = parser.parse_args()

    generate_dataset(
        n_samples=args.samples,
        output_path=args.output,
        seed=args.seed,
        val_split=args.val_split,
    )
