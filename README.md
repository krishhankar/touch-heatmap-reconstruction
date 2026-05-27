# touch-heatmap-reconstruction

## 1. Quick Intro
**Real-time end-to-end deep learning project**: Simulate a capacitive touch sensor from mouse input, train a UNet to denoise noisy 20×20 sensor frames, and watch predictions live in a PyGame window with speed-adaptive temporal smoothing.

## 2. What Problem This Solves & Why It's Unique
Capacitive touch sensors (used in trackpads, touchscreens, and robotic tactile skins) don't naturally output an `(X, Y)` coordinate. Instead, they output a **raw 2D grid of capacitance values** (a heatmap). In real hardware, this heatmap is incredibly messy due to baseline drift, electromagnetic interference (EMI), hardware imperfections, and ADC quantization noise.

If you try to calculate a cursor position directly from this raw, noisy hardware data, your cursor will jump around erratically. Traditional firmware uses complex heuristic algorithms and heavy Kalman filters to fix this, but they often struggle with non-linear noise or "ghost touches."

**What makes this unique:** 
Instead of relying on rigid, heuristic-based algorithms, this project uses an end-to-end Deep Learning pipeline to solve the problem. We train a lightweight, real-time convolutional neural network (GhostStrokeUNet) to "clean" the corrupted hardware heatmap, reconstructing a mathematically perfect touch blob. Combined with a sub-pixel soft-argmax extractor and a custom speed-adaptive exponential moving average (EMA) filter, it completely eliminates idle jitter while tracking rapid movements with zero lag.

## 3. Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt --break-system-packages

# 2. Generate synthetic dataset  (~2-3 min, ~35 MB)
python data/generate_dataset.py

# 3. Train the model  (~20-40 min on CPU)
python model/train.py

# 4. Run the live demo
python demo/live_demo.py
```

## 4. Project Structure

```text
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
│   └── live_demo.py          — 5-panel PyGame window (main entry point)
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

## 5. Architecture — GhostStrokeUNet

```text
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
