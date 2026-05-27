"""
model/train.py — Full training loop for GhostStrokeUNet.

Loads data/dataset.npz, trains for N epochs with Adam + ReduceLROnPlateau,
saves best checkpoint to checkpoints/ghoststroke_unet.pth, and plots
training curves to checkpoints/training_curves.png.

Usage:
    python model/train.py
    python model/train.py --epochs 50 --batch-size 128
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allow imports from project root regardless of CWD
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.unet import GhostStrokeUNet  # noqa: E402


# ── Dataset ───────────────────────────────────────────────────────────────────

class SensorFrameDataset(Dataset):
    """Wraps pre-computed (noisy, clean) numpy arrays as a PyTorch Dataset."""

    def __init__(self, noisy: np.ndarray, clean: np.ndarray) -> None:
        """Store tensors for __getitem__ access."""
        self.noisy = torch.tensor(noisy, dtype=torch.float32)
        self.clean = torch.tensor(clean, dtype=torch.float32)

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.noisy)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (noisy, clean) tensor pair at index idx."""
        return self.noisy[idx], self.clean[idx]


# ── Metrics ───────────────────────────────────────────────────────────────────

def centroid_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean Euclidean centroid error (pixels) over the batch.

    Uses differentiable soft-argmax (weighted average of grid indices).
    """
    B, C, H, W = pred.shape  # noqa: N806

    col_idx = torch.arange(W, dtype=torch.float32, device=pred.device)  # (W,)
    row_idx = torch.arange(H, dtype=torch.float32, device=pred.device)  # (H,)

    def soft_centroid(heatmap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (cx, cy) for a (B,1,H,W) heatmap tensor."""
        flat = heatmap.squeeze(1)  # (B, H, W)
        total = flat.sum(dim=[1, 2]).clamp(min=1e-8)  # (B,)
        cx = (flat * col_idx[None, None, :]).sum(dim=[1, 2]) / total
        cy = (flat * row_idx[None, :, None]).sum(dim=[1, 2]) / total
        return cx, cy

    pred_cx, pred_cy = soft_centroid(pred)
    true_cx, true_cy = soft_centroid(target)

    dist = torch.sqrt((pred_cx - true_cx) ** 2 + (pred_cy - true_cy) ** 2)
    return float(dist.mean().item())


# ── Training function ─────────────────────────────────────────────────────────

def train(
    dataset_path: str = "data/dataset.npz",
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    checkpoint_dir: str = "checkpoints",
    device_str: str = "auto",
) -> None:
    """Train GhostStrokeUNet and save the best checkpoint.

    Parameters
    ----------
    dataset_path   : path to the .npz file produced by generate_dataset.py
    epochs         : number of training epochs
    batch_size     : mini-batch size
    lr             : initial Adam learning rate
    checkpoint_dir : directory for saving checkpoint and curve plot
    device_str     : "auto" | "cpu" | "cuda"
    """
    # ── Device ────────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "ghoststroke_unet.pth")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset from {dataset_path} …")
    data = np.load(dataset_path)
    train_ds = SensorFrameDataset(data["train_noisy"], data["train_clean"])
    val_ds = SensorFrameDataset(data["val_noisy"], data["val_clean"])
    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=use_cuda,
    )

    # ── Model / loss / optimiser ───────────────────────────────────────────────
    model = GhostStrokeUNet().to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5,
    )

    # ── History ───────────────────────────────────────────────────────────────
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_centroid_mse": [],
    }
    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        # Train
        model.train()
        train_loss_acc = 0.0
        for noisy, clean in tqdm(
            train_loader,
            desc=f"Epoch {epoch:3d}/{epochs} [train]",
            leave=False,
            ncols=80,
        ):
            noisy = noisy.to(device)
            clean = clean.to(device)
            optimiser.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, clean)
            loss.backward()
            optimiser.step()
            train_loss_acc += loss.item() * noisy.size(0)

        train_loss = train_loss_acc / len(train_ds)

        # Validate
        model.eval()
        val_loss_acc = 0.0
        val_centroid_acc = 0.0
        with torch.no_grad():
            for noisy, clean in tqdm(
                val_loader,
                desc=f"Epoch {epoch:3d}/{epochs} [val]  ",
                leave=False,
                ncols=80,
            ):
                noisy = noisy.to(device)
                clean = clean.to(device)
                pred = model(noisy)
                val_loss_acc += criterion(pred, clean).item() * noisy.size(0)
                val_centroid_acc += centroid_mse(pred, clean) * noisy.size(0)

        val_loss = val_loss_acc / len(val_ds)
        val_centroid = val_centroid_acc / len(val_ds)
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_centroid_mse"].append(val_centroid)

        scheduler.step(val_loss)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimiser.state_dict(),
                    "val_loss": val_loss,
                    "val_centroid_mse": val_centroid,
                },
                ckpt_path,
            )
            marker = "  ★ saved"
        else:
            marker = ""

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"centroid={val_centroid:.3f}px  "
            f"({elapsed:.1f}s){marker}"
        )

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved to: {ckpt_path}")

    # ── Training curves ───────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history["train_loss"], label="Train loss", color="#2563EB")
        ax1.plot(history["val_loss"], label="Val loss", color="#DC2626")
        ax1.set_title("MSE Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MSE")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history["val_centroid_mse"], label="Val centroid error", color="#059669")
        ax2.axhline(0.5, color="#F59E0B", linestyle="--", label="0.5 px target")
        ax2.set_title("Centroid Error")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Error (pixels)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        curve_path = os.path.join(checkpoint_dir, "training_curves.png")
        plt.savefig(curve_path, dpi=150)
        plt.close(fig)
        print(f"Training curves saved to: {curve_path}")

    except ImportError as e:
        print(f"matplotlib not available, skipping curve plot: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GhostStrokeUNet.")
    parser.add_argument("--dataset", type=str,
                        default=os.path.join("data", "dataset.npz"),
                        help="Path to dataset.npz")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Number of epochs (default: 40)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint output directory (default: checkpoints)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto | cpu | cuda (default: auto)")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        device_str=args.device,
    )
