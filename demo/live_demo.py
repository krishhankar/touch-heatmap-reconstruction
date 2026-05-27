"""
demo/live_demo.py — Real-time PyGame touch-heatmap-reconstruction demo.

Four-panel window:
  [Noisy sensor]  [UNet prediction]
  [Live trajectory (ground truth vs predicted)]
  [Stats bar — FPS · latency · centroid XY]

Controls:
  Left-click + drag  Record trajectory
  R                  Clear trails
  S                  Save screenshot
  ESC / Q            Quit

Usage:
    python demo/live_demo.py
    python demo/live_demo.py --no-model
    python demo/live_demo.py --checkpoint checkpoints/ghoststroke_unet.pth
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from typing import Deque, List, Optional, Tuple

import numpy as np
import pygame
import torch

# Allow imports from project root regardless of CWD
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.unet import GhostStrokeUNet  # noqa: E402
from inference.centroid import (  # noqa: E402
    extract_centroid_from_tensor,
    frame_to_tensor,
    mouse_to_sensor_frame,
)

# ── Layout constants ──────────────────────────────────────────────────────────
GRID_SIZE: int = 20
PANEL_SIZE: int = 240
TRAJ_SIZE: int = 500
STATS_H: int = 50
PAD: int = 12

WINDOW_W: int = PANEL_SIZE * 2 + TRAJ_SIZE + PAD * 4
WINDOW_H: int = PANEL_SIZE + TRAJ_SIZE + STATS_H + PAD * 4

# ── Colour palette (RGB) ──────────────────────────────────────────────────────
BG_COLOR: Tuple[int, int, int] = (15, 17, 23)
PANEL_BG: Tuple[int, int, int] = (24, 28, 38)
BORDER_COLOR: Tuple[int, int, int] = (50, 60, 80)
TEXT_COLOR: Tuple[int, int, int] = (200, 210, 230)
ACCENT_BLUE: Tuple[int, int, int] = (37, 99, 235)
ACCENT_GREEN: Tuple[int, int, int] = (16, 185, 129)
ACCENT_RED: Tuple[int, int, int] = (239, 68, 68)
ACCENT_YELLOW: Tuple[int, int, int] = (245, 158, 11)
DIM_TEXT: Tuple[int, int, int] = (100, 115, 140)


# ── Colourmap ─────────────────────────────────────────────────────────────────

def apply_hot_colormap(frame: np.ndarray) -> np.ndarray:
    """Convert a (H, W) float32 frame in [0,1] to (H, W, 3) uint8 using the 'hot' map."""
    v = frame.astype(np.float32)
    r = np.clip(v * 3.0, 0.0, 1.0)
    g = np.clip(v * 3.0 - 1.0, 0.0, 1.0)
    b = np.clip(v * 3.0 - 2.0, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)  # (H, W, 3)
    return (rgb * 255).astype(np.uint8)


def frame_to_surface(frame_2d: np.ndarray, size: int) -> pygame.Surface:
    """Upscale a 20×20 heatmap to a (size, size) pygame.Surface.

    IMPORTANT: pygame.surfarray.make_surface expects (W, H, 3).
    Our RGB array is (H, W, 3), so we transpose axes (1, 0, 2).
    """
    rgb = apply_hot_colormap(frame_2d)      # (H, W, 3)
    rgb_t = rgb.transpose(1, 0, 2)         # (W, H, 3) — required by surfarray
    small_surf = pygame.surfarray.make_surface(rgb_t)
    return pygame.transform.scale(small_surf, (size, size))


# ── Panel helpers ─────────────────────────────────────────────────────────────

def draw_panel(
    screen: pygame.Surface,
    surface: pygame.Surface,
    x: int,
    y: int,
    label: str,
    font: pygame.font.Font,
) -> None:
    """Draw a bordered sensor panel with a label above it."""
    # Background
    panel_rect = pygame.Rect(x, y, PANEL_SIZE, PANEL_SIZE)
    pygame.draw.rect(screen, PANEL_BG, panel_rect)
    pygame.draw.rect(screen, BORDER_COLOR, panel_rect, 2)

    # Content
    screen.blit(surface, (x, y))

    # Label above panel
    label_surf = font.render(label, True, TEXT_COLOR)
    screen.blit(label_surf, (x, y - label_surf.get_height() - 3))


def draw_trajectory(
    screen: pygame.Surface,
    true_trail: Deque[Tuple[float, float]],
    pred_trail: Deque[Tuple[float, float]],
    x: int,
    y: int,
    w: int,
    h: int,
    font: pygame.font.Font,
) -> None:
    """Draw the ground-truth (green) and predicted (red) trajectory panel."""
    # Panel background
    traj_rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, PANEL_BG, traj_rect)
    pygame.draw.rect(screen, BORDER_COLOR, traj_rect, 2)

    # Light grid lines every 5 grid cells
    step_x = w / GRID_SIZE
    step_y = h / GRID_SIZE
    for col in range(0, GRID_SIZE + 1, 5):
        gx = x + int(col * step_x)
        pygame.draw.line(screen, BORDER_COLOR, (gx, y), (gx, y + h))
    for row in range(0, GRID_SIZE + 1, 5):
        gy = y + int(row * step_y)
        pygame.draw.line(screen, BORDER_COLOR, (x, gy), (x + w, gy))

    def to_screen(cx: float, cy: float) -> Tuple[int, int]:
        """Map grid coords to screen pixel coords."""
        sx = int(x + cx / GRID_SIZE * w)
        sy = int(y + cy / GRID_SIZE * h)
        return sx, sy

    # Ground-truth trail (green)
    if len(true_trail) > 1:
        pts = [to_screen(cx, cy) for cx, cy in true_trail]
        pygame.draw.lines(screen, ACCENT_GREEN, False, pts, 2)
    if true_trail:
        last = to_screen(*true_trail[-1])
        pygame.draw.circle(screen, ACCENT_GREEN, last, 5)

    # Predicted trail (red)
    if len(pred_trail) > 1:
        pts = [to_screen(cx, cy) for cx, cy in pred_trail]
        pygame.draw.lines(screen, ACCENT_RED, False, pts, 2)
    if pred_trail:
        last = to_screen(*pred_trail[-1])
        pygame.draw.circle(screen, ACCENT_RED, last, 4)

    # Legend
    font_s = pygame.font.SysFont("Arial", 12)
    screen.blit(font_s.render("● True", True, ACCENT_GREEN),
                (x + 6, y + h - 18))
    screen.blit(font_s.render("● Pred", True, ACCENT_RED),
                (x + 60, y + h - 18))


def draw_stats(
    screen: pygame.Surface,
    fps: float,
    inference_ms: float,
    true_xy: Tuple[float, float],
    pred_xy: Tuple[float, float],
    font: pygame.font.Font,
    font_small: pygame.font.Font,
    using_model: bool,
) -> None:
    """Render the stats bar at the bottom of the window."""
    bar_y = WINDOW_H - STATS_H
    stats_rect = pygame.Rect(0, bar_y, WINDOW_W, STATS_H)
    pygame.draw.rect(screen, (18, 22, 32), stats_rect)
    pygame.draw.line(screen, BORDER_COLOR, (0, bar_y), (WINDOW_W, bar_y), 1)

    # FPS — colour coded
    fps_color = ACCENT_GREEN if fps >= 60 else (ACCENT_YELLOW if fps >= 30 else ACCENT_RED)

    items = [
        (f"touch-heatmap-reconstruction", TEXT_COLOR, 12, True),
        (f"FPS {fps:5.1f}", fps_color, 150, False),
        (f"Latency {inference_ms:.1f}ms", TEXT_COLOR, 270, False),
        (f"True ({true_xy[0]:.1f}, {true_xy[1]:.1f})", ACCENT_GREEN, 420, False),
        (f"Pred ({pred_xy[0]:.1f}, {pred_xy[1]:.1f})", ACCENT_RED, 590, False),
    ]
    for text, color, offset, bold in items:
        f = font if bold else font_small
        surf = f.render(text, True, color)
        screen.blit(surf, (offset, bar_y + (STATS_H - surf.get_height()) // 2))

    # Hint line
    mode_str = "MODEL" if using_model else "NO-MODEL (fallback)"
    hint = (
        f"Mode: {mode_str}  |  "
        "Left-drag: draw  |  R: clear  |  S: screenshot  |  Q/ESC: quit"
    )
    hint_surf = font_small.render(hint, True, DIM_TEXT)
    screen.blit(hint_surf, (12, bar_y + STATS_H - hint_surf.get_height() - 4))


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_demo(
    checkpoint_path: str,
    no_model: bool,
) -> None:
    """Run the PyGame live demo window."""
    pygame.init()
    pygame.display.set_caption("touch-heatmap-reconstruction")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Arial", 15, bold=True)
    font_small = pygame.font.SysFont("Arial", 13)

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: Optional[GhostStrokeUNet] = None
    using_model = False

    if not no_model:
        if os.path.exists(checkpoint_path):
            try:
                m = GhostStrokeUNet().to(device)
                ckpt = torch.load(
                    checkpoint_path, map_location=device, weights_only=True
                )
                m.load_state_dict(ckpt["model_state_dict"])
                m.eval()
                model = m
                using_model = True
                print(
                    f"Model loaded from {checkpoint_path} "
                    f"(epoch {ckpt['epoch']})"
                )
            except Exception as e:
                print(f"WARNING: Could not load model: {e}")
                print("Running in fallback (no-model) mode.")
        else:
            print(
                f"WARNING: checkpoint not found at {checkpoint_path}. "
                "Run `python model/train.py` first, or use --no-model flag."
            )
            print("Running in fallback (no-model) mode.")

    # ── State ─────────────────────────────────────────────────────────────────
    TRAIL_MAX = 2000
    true_trail: Deque[Tuple[float, float]] = collections.deque(maxlen=TRAIL_MAX)
    pred_trail: Deque[Tuple[float, float]] = collections.deque(maxlen=TRAIL_MAX)
    is_drawing = False
    inference_ms = 0.0
    screenshot_count = 0

    # Panel anchor positions
    panel_noisy_x = PAD
    panel_noisy_y = PAD + 20
    panel_pred_x = PAD * 2 + PANEL_SIZE
    panel_pred_y = PAD + 20
    traj_x = PAD * 3 + PANEL_SIZE * 2
    traj_y = PAD + 20
    traj_w = TRAJ_SIZE
    traj_h = PANEL_SIZE + TRAJ_SIZE - 20 + PAD

    # Placeholder blank surface
    blank_surf = frame_to_surface(np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32), PANEL_SIZE)

    running = True
    while running:
        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    true_trail.clear()
                    pred_trail.clear()
                elif event.key == pygame.K_s:
                    screenshot_count += 1
                    fname = f"screenshot_{screenshot_count:03d}.png"
                    pygame.image.save(screen, fname)
                    print(f"Screenshot saved: {fname}")

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                is_drawing = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                is_drawing = False

        # ── Per-frame computation ─────────────────────────────────────────────
        mouse_x, mouse_y = pygame.mouse.get_pos()

        # Ground-truth grid position (map full window coords)
        gt_cx = (mouse_x / WINDOW_W) * GRID_SIZE
        gt_cy = (mouse_y / WINDOW_H) * GRID_SIZE
        gt_cx = float(np.clip(gt_cx, 0.0, GRID_SIZE - 1.0))
        gt_cy = float(np.clip(gt_cy, 0.0, GRID_SIZE - 1.0))

        # Sensor frame from mouse position
        frame = mouse_to_sensor_frame(mouse_x, mouse_y, WINDOW_W, WINDOW_H)
        noisy_2d = frame[0]  # (H, W)

        # Inference
        t_start = time.perf_counter()
        if model is not None:
            tensor_in = frame_to_tensor(frame, device)
            with torch.no_grad():
                pred_tensor = model(tensor_in)
            pred_cx, pred_cy = extract_centroid_from_tensor(pred_tensor)
            pred_2d = pred_tensor.detach().cpu().squeeze().numpy().astype(np.float32)
        else:
            pred_cx, pred_cy = gt_cx, gt_cy
            pred_2d = noisy_2d.copy()
        inference_ms = (time.perf_counter() - t_start) * 1000.0

        # Record trail
        if is_drawing:
            true_trail.append((gt_cx, gt_cy))
            pred_trail.append((pred_cx, pred_cy))

        # ── Render ────────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # Convert frames to surfaces
        noisy_surf = frame_to_surface(noisy_2d, PANEL_SIZE)
        pred_surf = frame_to_surface(pred_2d, PANEL_SIZE)

        # Draw sensor panels
        draw_panel(screen, noisy_surf, panel_noisy_x, panel_noisy_y,
                   "Raw Noisy Sensor", font)
        draw_panel(screen, pred_surf, panel_pred_x, panel_pred_y,
                   "UNet Prediction" if using_model else "Noisy (no model)", font)

        # Trajectory panel (extends below panels)
        draw_trajectory(screen, true_trail, pred_trail,
                        traj_x, traj_y, traj_w, traj_h, font)
        traj_label = font.render("Trajectory", True, TEXT_COLOR)
        screen.blit(traj_label, (traj_x, traj_y - traj_label.get_height() - 3))

        # Stats bar
        fps = clock.get_fps()
        draw_stats(
            screen, fps, inference_ms,
            (gt_cx, gt_cy), (pred_cx, pred_cy),
            font, font_small, using_model,
        )

        # Crosshair cursor
        cx_c, cy_c = mouse_x, mouse_y
        pygame.draw.line(screen, TEXT_COLOR, (cx_c - 10, cy_c), (cx_c + 10, cy_c), 1)
        pygame.draw.line(screen, TEXT_COLOR, (cx_c, cy_c - 10), (cx_c, cy_c + 10), 1)

        pygame.display.flip()
        clock.tick(120)

    pygame.quit()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="touch-heatmap-reconstruction — live PyGame demo"
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="Run without loading a checkpoint (raw sensor display only)",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.join("checkpoints", "ghoststroke_unet.pth"),
        help="Path to checkpoint file (default: checkpoints/ghoststroke_unet.pth)",
    )
    args = parser.parse_args()

    run_demo(
        checkpoint_path=args.checkpoint,
        no_model=args.no_model,
    )
