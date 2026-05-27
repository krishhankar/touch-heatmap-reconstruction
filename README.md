# Touch Heatmap Reconstruction

A deep learning project that reconstructs clean sub-pixel touch coordinates from extremely noisy capacitive sensor heatmaps. It uses a lightweight PyTorch UNet to filter out hardware interference and a speed-adaptive algorithm to provide ultra-smooth, real-time trajectory tracking.

---

## 🚀 Quick Start

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Generate the Synthetic Dataset:**
```bash
python data/generate_dataset.py --samples 50000
```

3. **Train the UNet Model:**
```bash
python model/train.py --epochs 40 --batch-size 64
```

4. **Run the Live Demo:**
```bash
# Draw with Left-Click. Press 'R' to clear, 'S' to save screenshot, 'ESC' to quit.
python demo/live_demo.py
```

---

## 📁 Project Structure

```text
touch-heatmap-reconstruction/
├── data/
│   └── generate_dataset.py   # Synthetic noise & gaussian blob generator
├── model/
│   ├── unet.py               # GhostStrokeUNet architecture
│   └── train.py              # PyTorch training loop
├── inference/
│   └── centroid.py           # Sub-pixel Soft-Argmax logic
├── demo/
│   └── live_demo.py          # Real-time PyGame trajectory visualization
├── utils/
│   └── noise.py              # Hardware noise simulation (Drift, EMI, etc.)
└── checkpoints/              # Saved model weights & training curves
```

---

## 🧠 Architecture

The system operates in a four-stage pipeline:
1. **Noise Simulation:** A clean 2D Gaussian touch is corrupted with baseline drift, Gaussian noise, salt-and-pepper artifacts, and ADC quantization to simulate physical hardware.
2. **Denoising UNet:** A lightweight encoder-decoder CNN (`GhostStrokeUNet`) processes the 20x20 noisy grid and reconstructs a mathematically clean probability heatmap.
3. **Soft-Argmax Centroid:** Instead of snapping to the brightest integer pixel, a spatial weighted average extracts the exact `(X, Y)` coordinate with sub-pixel precision.
4. **Speed-Adaptive Smoothing:** A dynamic Exponential Moving Average (EMA) filter heavily smooths slow movements to eliminate idle jitter, but reduces smoothing during fast movements for zero-lag responsiveness.

---

## 🎯 What Problem Does This Solve?

### The Problem
Capacitive touch sensors (like trackpads or robotic skins) output raw grids of capacitance values. In the real world, these heatmaps are plagued by temperature drift, electrical interference, and dead pixels. Traditional firmware uses heavy heuristic algorithms or Kalman filters to derive a cursor position. However, these classical methods often fail on non-linear noise, cause severe cursor jitter, or introduce heavy rubber-band lag.

### What Makes This Unique?
Instead of manually tuning complex math filters, this project treats sensor tracking as an **image-to-image translation problem**. 
- **AI-Driven Denoising:** The UNet organically learns what a "true touch" looks like versus background interference.
- **Sub-Pixel Accuracy:** The soft-argmax technique provides vastly higher resolution than the physical sensor grid allows.
- **Zero-Lag Smoothing:** The speed-adaptive tracker ensures the cursor feels perfectly steady when drawing slowly, but instantly responsive when swiping fast.
