import time
import numpy as np
import cv2
import sys
import torch
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from reel_parallax import render_parallax_frame, estimate_depth

def benchmark():
    h, w = 1920, 1080
    img = np.zeros((h, w, 3), dtype=np.uint8)

    print("Benchmarking depth estimation...")
    # MiDaS download might take a while on first run
    try:
        start_depth = time.time()
        depth = estimate_depth(img)
        end_depth = time.time()
        print(f"Depth estimation took: {end_depth - start_depth:.2f}s")
    except Exception as e:
        print(f"Depth estimation failed (maybe no internet or model download issue): {e}")
        depth = np.random.rand(h, w).astype(np.float32)

    zoom = 1.05
    dx_px = 5.0
    dy_px = 2.0
    depth_scale = 0.3

    iters = 100

    print(f"Benchmarking rendering ({iters} frames, scale={depth_scale})...")
    # Pre-compute parallax map as done in scripts/reel_parallax.py
    parallax = (1.0 - depth_scale) + depth_scale * depth

    # Warmup
    _ = render_parallax_frame(img, parallax, zoom, dx_px, dy_px)

    start = time.time()
    for _ in range(iters):
        _ = render_parallax_frame(img, parallax, zoom, dx_px, dy_px)
    total_time = time.time() - start

    print(f"Average time per frame: {total_time/iters*1000:.2f} ms")

    print(f"Benchmarking rendering ({iters} frames, scale=0.0)...")
    depth_scale_zero = 0.0
    parallax_zero = (1.0 - depth_scale_zero) + depth_scale_zero * depth
    start = time.time()
    for _ in range(iters):
        _ = render_parallax_frame(img, parallax_zero, zoom, dx_px, dy_px)
    total_time = time.time() - start
    print(f"Average time per frame (scale=0.0): {total_time/iters*1000:.2f} ms")

if __name__ == "__main__":
    benchmark()
