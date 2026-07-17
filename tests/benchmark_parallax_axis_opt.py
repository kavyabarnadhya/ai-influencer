import time
import numpy as np
import cv2
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from reel_parallax import render_parallax_frame

def render_parallax_frame_reference_implementation(
    img_bgr: np.ndarray,
    parallax: np.ndarray,
    zoom: float,
    dx_px: float,
    dy_px: float,
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    xs_centered = np.arange(w, dtype=np.float32) - cx
    ys_centered = np.arange(h, dtype=np.float32) - cy
    inv_z = 1.0 / zoom
    src_x = parallax * (-dx_px)
    src_x += (xs_centered * inv_z + cx)
    src_y = parallax * (-dy_px)
    src_y += (ys_centered.reshape(-1, 1) * inv_z + cy)
    return cv2.remap(
        img_bgr, src_x, src_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

def run_benchmark():
    h, w = 1920, 1080
    img = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.random.rand(h, w).astype(np.float32)

    zoom = 1.05
    depth_scale = 0.3
    parallax = (1.0 - depth_scale) + depth_scale * depth

    iters = 30

    print(f"--- Parallax Axis Optimization Benchmark ({w}x{h}, {iters} iterations) ---")

    # Scenario A: BOTH translations zero (dx=0, dy=0) -> should bypass maps & remap entirely
    print("\n[Scenario A: dx=0, dy=0 (Pure zoom, very common in Ken Burns default mode)]")
    start = time.perf_counter()
    for _ in range(iters):
        _ = render_parallax_frame_reference_implementation(img, parallax, zoom, 0.0, 0.0)
    ref_time_a = (time.perf_counter() - start) / iters

    start = time.perf_counter()
    for _ in range(iters):
        _ = render_parallax_frame(img, parallax, zoom, 0.0, 0.0)
    opt_time_a = (time.perf_counter() - start) / iters
    print(f"  Reference Implementation: {ref_time_a*1000:.2f} ms")
    print(f"  Optimized Implementation: {opt_time_a*1000:.2f} ms")
    print(f"  Speedup: {ref_time_a/opt_time_a:.2f}x (Saves {(ref_time_a - opt_time_a)*1000:.2f} ms per frame)")

    # Scenario B: ONE translation zero (dx=5, dy=0) -> should use np.broadcast_to for y-axis
    print("\n[Scenario B: dx=5, dy=0 (Horizontal sway only, extremely common)]")
    start = time.perf_counter()
    for _ in range(iters):
        _ = render_parallax_frame_reference_implementation(img, parallax, zoom, 5.0, 0.0)
    ref_time_b = (time.perf_counter() - start) / iters

    start = time.perf_counter()
    for _ in range(iters):
        _ = render_parallax_frame(img, parallax, zoom, 5.0, 0.0)
    opt_time_b = (time.perf_counter() - start) / iters
    print(f"  Reference Implementation: {ref_time_b*1000:.2f} ms")
    print(f"  Optimized Implementation: {opt_time_b*1000:.2f} ms")
    print(f"  Speedup: {ref_time_b/opt_time_b:.2f}x (Saves {(ref_time_b - opt_time_b)*1000:.2f} ms per frame)")

    # Scenario C: BOTH translations non-zero (dx=5, dy=2) -> standard dynamic path
    print("\n[Scenario C: dx=5, dy=2 (Full 2.5D sway + dolly)]")
    start = time.perf_counter()
    for _ in range(iters):
        _ = render_parallax_frame_reference_implementation(img, parallax, zoom, 5.0, 2.0)
    ref_time_c = (time.perf_counter() - start) / iters

    start = time.perf_counter()
    for _ in range(iters):
        _ = render_parallax_frame(img, parallax, zoom, 5.0, 2.0)
    opt_time_c = (time.perf_counter() - start) / iters
    print(f"  Reference Implementation: {ref_time_c*1000:.2f} ms")
    print(f"  Optimized Implementation: {opt_time_c*1000:.2f} ms")
    print(f"  Speedup: {ref_time_c/opt_time_c:.2f}x")

    # Document slide level savings (e.g. for a 150-frame video)
    print("\n--- Wall-Clock Cumulative Impact (150-frame reel) ---")
    saves_a = (ref_time_a - opt_time_a) * 150
    saves_b = (ref_time_b - opt_time_b) * 150
    print(f"  For default mode (pure zoom/dolly): saves {saves_a:.2f}s of render time per reel!")
    print(f"  For horizontal-sway mode: saves {saves_b:.2f}s of render time per reel!")

if __name__ == "__main__":
    run_benchmark()
