import numpy as np
import cv2
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from reel_parallax import render_parallax_frame, _GRID_CACHE

def render_parallax_frame_reference(
    img_bgr: np.ndarray,
    depth: np.ndarray,
    zoom: float,
    dx_px: float,
    dy_px: float,
    depth_scale: float,
) -> np.ndarray:
    """Original unoptimized logic for reference."""
    h, w = img_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    parallax = (1.0 - depth_scale) + depth_scale * depth
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    inv_z = 1.0 / zoom
    src_x = (grid_x - cx) * inv_z + cx
    src_y = (grid_y - cy) * inv_z + cy
    src_x = src_x - dx_px * parallax
    src_y = src_y - dy_px * parallax
    return cv2.remap(
        img_bgr, src_x, src_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

def test_render_parallax_frame_correctness():
    np.random.seed(42)
    h, w = 128, 64  # Small size for quick test
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    depth = np.random.rand(h, w).astype(np.float32)

    zoom = 1.05
    dx_px = 5.0
    dy_px = 2.0
    depth_scale = 0.3

    # Clear cache to ensure we test the grid creation too
    _GRID_CACHE.clear()

    # Pre-compute parallax for optimized call
    parallax = (1.0 - depth_scale) + depth_scale * depth

    # Call optimized
    optimized_out = render_parallax_frame(img, parallax, zoom, dx_px, dy_px)

    # Check that grid was cached
    assert (h, w) in _GRID_CACHE

    # Call reference
    reference_out = render_parallax_frame_reference(img, depth, zoom, dx_px, dy_px, depth_scale)

    # Compare results. INTER_LINEAR might have small differences due to floating point.
    diff = np.abs(optimized_out.astype(np.float32) - reference_out.astype(np.float32))
    print(f"Max diff: {np.max(diff)}")
    print(f"Mean diff: {np.mean(diff)}")
    assert np.max(diff) < 5, f"Max diff {np.max(diff)} exceeds threshold"
    assert np.mean(diff) < 0.1, f"Mean diff {np.mean(diff)} exceeds threshold"

    # Second call should use cache
    # Verify that it produces the same result as the first optimized call
    cached_out = render_parallax_frame(img, parallax, zoom, dx_px, dy_px)
    assert np.array_equal(cached_out, optimized_out)

    print("Correctness test passed.")

if __name__ == "__main__":
    try:
        test_render_parallax_frame_correctness()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
