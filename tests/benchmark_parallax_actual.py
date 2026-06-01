import time
import numpy as np
import cv2
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from reel_parallax import render_parallax_frame

def benchmark():
    h, w = 1920, 1080
    img = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.random.rand(h, w).astype(np.float32)

    zoom = 1.05
    dx_px = 5.0
    dy_px = 2.0
    depth_scale = 0.3

    iters = 50

    # Warmup
    _ = render_parallax_frame(img, depth, zoom, dx_px, dy_px, depth_scale)

    start = time.time()
    for _ in range(iters):
        _ = render_parallax_frame(img, depth, zoom, dx_px, dy_px, depth_scale)
    total_time = time.time() - start

    print(f"Average time per frame (actual code): {total_time/iters*1000:.2f} ms")

if __name__ == "__main__":
    benchmark()
