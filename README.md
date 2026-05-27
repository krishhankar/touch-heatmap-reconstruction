# touch-heatmap-reconstruction

![Python](https://img.shields.io/badge/python-3.10.20-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Real-time end-to-end deep learning project**: simulate a capacitive touch sensor from
mouse input, train a UNet to denoise noisy 20×20 sensor frames, and watch predictions live
in a four-panel PyGame window.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt --break-system-packages

# 2. Generate synthetic dataset  (~2-3 min, ~35 MB)
python data/generate_dataset.py

# 3. Train the model  (~20-40 min on CPU)
python model/train.py

# 4. (Optional) Evaluate and generate comparison images
python model/evaluate.py

# 5. Run the live demo
python demo/live_demo.py

# 5b. Run live demo without a trained model (raw sensor only)
python demo/live_demo.py --no-model
```

---

## Controls

| Key / Action        | Effect                              |
|---------------------|-------------------------------------|
| Left-click + drag   | Record trajectory                   |
| `R`                 | Clear both trails                   |
| `S`                 | Save screenshot (`screenshot_NNN.png`) |
| `ESC` / `Q`         | Quit                                |

---

## Project Structure

```
touch-heatmap-reconstruction/
├── data/
│   └── generate_dataset.py   — synthetic (noisy, clean) pair generator
├── model/
│   ├── unet.py               — GhostStrokeUNet architecture
│   ├── train.py              — training loop + checkpoint + curves
│   └── evaluate.py           — metrics + visual comparison grid
├── inference/
│   └── centroid.py           — soft/hard argmax + frame utilities
├── demo/
│   └── live_demo.py          — 4-panel PyGame window (main entry point)
├── utils/
│   ├── __init__.py
│   └── noise.py              — gaussian / salt&pepper / drift / quantization
├── checkpoints/              — created by train.py
│   ├── ghoststroke_unet.pth
│   ├── training_curves.png
│   └── evaluation.png
├── __init__.py
├── requirements.txt
└── README.md
```

---

## Architecture — GhostStrokeUNet

```
Input  (B, 1, 20, 20)  — noisy sensor frame
  │
  ├─ Encoder
  │    enc1  ConvBlock(1  → 16)   → (B, 16, 20, 20)
  │    pool1 MaxPool2d(2)         → (B, 16, 10, 10)
  │    enc2  ConvBlock(16 → 32)   → (B, 32, 10, 10)
  │    pool2 MaxPool2d(2)         → (B, 32,  5,  5)
  │
  ├─ Bottleneck
  │    ConvBlock(32 → 64)         → (B, 64,  5,  5)
  │
  ├─ Decoder  (skip connections)
  │    up2   ConvTranspose2d(64→32)  → (B, 32, 10, 10)
  │    dec2  ConvBlock(64 → 32)      (concat with enc2)
  │    up1   ConvTranspose2d(32→16)  → (B, 16, 20, 20)
  │    dec1  ConvBlock(32 → 16)      (concat with enc1)
  │
  └─ Head  Conv2d(16→1) + Sigmoid → (B, 1, 20, 20)

Trainable parameters: ~58,000
```

---

## Noise Pipeline (`utils/noise.py`)

Applied in order to every clean Gaussian heatmap:

1. **Baseline drift** — sinusoidal 2-D low-frequency surface
2. **Gaussian noise** — σ ∈ U(0.05, 0.15)
3. **Salt & pepper** — ratio ∈ U(0.01, 0.05)
4. **Quantization** — 8-bit (256 levels)

---

## Execution Order

| Step | Command | Time |
|------|---------|------|
| Install | `pip install -r requirements.txt --break-system-packages` | ~2 min |
| Dataset | `python data/generate_dataset.py` | ~2-3 min |
| Train | `python model/train.py` | ~20-40 min (CPU) |
| Evaluate | `python model/evaluate.py` | ~30 sec |
| Demo | `python demo/live_demo.py` | instant |

---

## Requirements

```
torch>=2.0.0,<2.4.0
numpy>=1.24.0,<2.0.0
pygame>=2.5.0
matplotlib>=3.7.0
scipy>=1.11.0
tqdm>=4.65.0
```

Python 3.10.20 · Linux · CPU-only compatible
