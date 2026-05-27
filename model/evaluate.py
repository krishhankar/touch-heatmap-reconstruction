"""
model/evaluate.py — Post-training evaluation with metrics and visual comparison.

Loads checkpoints/ghoststroke_unet.pth and evaluates it on the validation
split of data/dataset.npz.  Saves a grid comparison image to
checkpoints/evaluation.png.

Usage:
    python model/evaluate.py
    python model/evaluate.py --samples 12
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Allow imports from project root regardless of CWD
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.unet import GhostStrokeUNet  # noqa: E402
from inference.centroid import soft_argmax  # noqa: E402


def evaluate(
    dataset_path: str = "data/dataset.npz",
    checkpoint_path: str = os.path.join("checkpoints", "ghoststroke_unet.pth"),
    n_display: int = 8,
    device_str: str = "auto",
) -> None:
    """Run full evaluation and save evaluation.png.

    Parameters
    ----------
    dataset_path    : path to the .npz dataset
    checkpoint_path : path to the saved model checkpoint
    n_display       : number of samples to display in the comparison grid
    device_str      : "auto" | "cpu" | "cuda"
    """
    # ── Device ────────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found at {checkpoint_path}")
        print("Run `python model/train.py` first.")
        return

    model = GhostStrokeUNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"Loaded checkpoint (epoch {ckpt['epoch']}, "
        f"val_loss={ckpt['val_loss']:.6f})"
    )

    # ── Load val data ─────────────────────────────────────────────────────────
    data = np.load(dataset_path)
    val_noisy = data["val_noisy"]   # (N, 1, H, W)
    val_clean = data["val_clean"]
    val_pos = data["val_pos"]       # (N, 2)  [cx, cy]
    n_val = len(val_noisy)
    print(f"Validation samples: {n_val:,}")

    # ── Inference loop ────────────────────────────────────────────────────────
    criterion = nn.MSELoss(reduction="mean")
    pred_cx_list: List[float] = []
    pred_cy_list: List[float] = []
    true_cx_list: List[float] = []
    true_cy_list: List[float] = []
    mse_list: List[float] = []

    batch_size = 256
    pred_heatmaps: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, n_val, batch_size):
            end = min(start + batch_size, n_val)
            noisy_batch = torch.tensor(val_noisy[start:end], dtype=torch.float32).to(device)
            clean_batch = torch.tensor(val_clean[start:end], dtype=torch.float32).to(device)

            pred_batch = model(noisy_batch)  # (B, 1, H, W)
            mse_list.append(criterion(pred_batch, clean_batch).item())

            pred_np = pred_batch.cpu().numpy()  # (B, 1, H, W)
            pred_heatmaps.append(pred_np)

            for j in range(pred_np.shape[0]):
                px, py = soft_argmax(pred_np[j, 0])
                pred_cx_list.append(px)
                pred_cy_list.append(py)

            for j in range(end - start):
                true_cx_list.append(float(val_pos[start + j, 0]))
                true_cy_list.append(float(val_pos[start + j, 1]))

    pred_cx = np.array(pred_cx_list, dtype=np.float32)
    pred_cy = np.array(pred_cy_list, dtype=np.float32)
    true_cx = np.array(true_cx_list, dtype=np.float32)
    true_cy = np.array(true_cy_list, dtype=np.float32)

    # ── Metrics ───────────────────────────────────────────────────────────────
    errors = np.sqrt((pred_cx - true_cx) ** 2 + (pred_cy - true_cy) ** 2)
    mae_x = float(np.abs(pred_cx - true_cx).mean())
    mae_y = float(np.abs(pred_cy - true_cy).mean())
    mean_err = float(errors.mean())
    median_err = float(np.median(errors))
    pct_05 = float((errors < 0.5).mean() * 100)
    pct_10 = float((errors < 1.0).mean() * 100)
    mean_mse = float(np.mean(mse_list))

    print("\n── Evaluation Metrics ──────────────────────────────────")
    print(f"  Mean centroid error  : {mean_err:.4f} px")
    print(f"  Median centroid error: {median_err:.4f} px")
    print(f"  MAE X                : {mae_x:.4f} px")
    print(f"  MAE Y                : {mae_y:.4f} px")
    print(f"  Error < 0.5 px       : {pct_05:.1f}%  (target: >80%)")
    print(f"  Error < 1.0 px       : {pct_10:.1f}%")
    print(f"  Mean heatmap MSE     : {mean_mse:.6f}")
    print("────────────────────────────────────────────────────────\n")

    # ── Visual comparison grid ────────────────────────────────────────────────
    # Gather n_display samples
    indices = np.arange(min(n_display, n_val))
    all_preds = np.concatenate(pred_heatmaps, axis=0)  # (N_val, 1, H, W)

    n_cols = len(indices)
    fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 2.0, 6.5))
    fig.patch.set_facecolor("#0F1117")

    row_labels = ["Noisy Input", "UNet Prediction", "Ground Truth"]
    for col, idx in enumerate(indices):
        noisy_img = val_noisy[idx, 0]    # (H, W)
        pred_img = all_preds[idx, 0]     # (H, W)
        clean_img = val_clean[idx, 0]    # (H, W)

        px, py = soft_argmax(pred_img)
        tx, ty = float(val_pos[idx, 0]), float(val_pos[idx, 1])

        for row, (img, marker, mcolor) in enumerate([
            (noisy_img, None, None),
            (pred_img, (px, py), "cyan"),
            (clean_img, (tx, ty), "lime"),
        ]):
            ax = axes[row, col] if n_cols > 1 else axes[row]
            ax.imshow(img, cmap="hot", vmin=0, vmax=1, interpolation="nearest")
            if marker is not None:
                ax.plot(marker[0], marker[1], "+", color=mcolor,
                        markersize=10, markeredgewidth=2)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_labels[row], color="white", fontsize=9)
            ax.spines["bottom"].set_color("#3C4563")
            ax.spines["top"].set_color("#3C4563")
            ax.spines["left"].set_color("#3C4563")
            ax.spines["right"].set_color("#3C4563")

    plt.suptitle(
        "touch-heatmap-reconstruction — Evaluation",
        color="white", fontsize=13, y=1.01,
    )
    plt.tight_layout()

    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    os.makedirs(ckpt_dir, exist_ok=True)
    out_path = os.path.join(ckpt_dir, "evaluation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    print(f"Evaluation grid saved to: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GhostStrokeUNet.")
    parser.add_argument("--dataset", type=str,
                        default=os.path.join("data", "dataset.npz"),
                        help="Path to dataset.npz")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join("checkpoints", "ghoststroke_unet.pth"),
                        help="Path to checkpoint .pth file")
    parser.add_argument("--samples", type=int, default=8,
                        help="Number of samples to display in comparison grid (default: 8)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto | cpu | cuda (default: auto)")
    args = parser.parse_args()

    evaluate(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        n_display=args.samples,
        device_str=args.device,
    )
