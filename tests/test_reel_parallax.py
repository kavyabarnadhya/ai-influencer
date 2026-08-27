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

    # Pre-calculate parallax map for optimized version
    parallax = (1.0 - depth_scale) + depth_scale * depth

    # Call optimized
    optimized_out = render_parallax_frame(img, parallax, zoom, dx_px, dy_px)

    # Check that grid was cached
    assert (h, w) in _GRID_CACHE

    # Call reference
    reference_out = render_parallax_frame_reference(img, depth, zoom, dx_px, dy_px, depth_scale)

    # Compare results. INTER_LINEAR might have small differences due to floating point.
    # Max pixel difference of 3 is actually quite small in 0-255 range.
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


def test_render_parallax_frame_axis_optimizations():
    np.random.seed(42)
    h, w = 128, 64  # Small size for quick test
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    depth = np.random.rand(h, w).astype(np.float32)

    zoom = 1.05
    depth_scale = 0.3
    parallax = (1.0 - depth_scale) + depth_scale * depth

    # Case 1: dx_px == 0 and dy_px == 0
    dx_px, dy_px = 0.0, 0.0
    opt_out = render_parallax_frame(img, parallax, zoom, dx_px, dy_px)
    ref_out = render_parallax_frame_reference(img, depth, zoom, dx_px, dy_px, depth_scale)
    assert np.max(np.abs(opt_out.astype(np.float32) - ref_out.astype(np.float32))) < 5

    # Case 2: dx_px == 0 but dy_px != 0
    dx_px, dy_px = 0.0, 3.0
    opt_out = render_parallax_frame(img, parallax, zoom, dx_px, dy_px)
    ref_out = render_parallax_frame_reference(img, depth, zoom, dx_px, dy_px, depth_scale)
    assert np.max(np.abs(opt_out.astype(np.float32) - ref_out.astype(np.float32))) < 5

    # Case 3: dx_px != 0 but dy_px == 0
    dx_px, dy_px = 3.0, 0.0
    opt_out = render_parallax_frame(img, parallax, zoom, dx_px, dy_px)
    ref_out = render_parallax_frame_reference(img, depth, zoom, dx_px, dy_px, depth_scale)
    assert np.max(np.abs(opt_out.astype(np.float32) - ref_out.astype(np.float32))) < 5


def test_estimate_depth_caching(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip("torch")
    import torch
    from reel_parallax import _load_midas, estimate_depth

    # Create a dummy image file and read it
    img_path = tmp_path / "slide_test.png"
    img_bgr = np.zeros((128, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img_bgr)

    midas_calls = 0
    def mock_load_midas():
        nonlocal midas_calls
        midas_calls += 1
        # Model returns a 3D tensor
        model = lambda x: torch.zeros((1, 32, 16), dtype=torch.float32)
        transform = lambda x: torch.zeros((1, 3, 32, 16), dtype=torch.float32)
        device = torch.device("cpu")
        return model, transform, device

    _load_midas.cache_clear()
    monkeypatch.setattr("reel_parallax._load_midas", mock_load_midas)

    # First call: Cache miss, should trigger MiDaS load and run
    depth1 = estimate_depth(img_bgr, smooth_sigma=0.0, img_path=img_path)
    assert midas_calls == 1
    assert depth1.shape == (128, 64)

    cache_path = tmp_path / "slide_test.png.depth.npz"
    assert cache_path.exists()

    # Second call: Cache hit, should load from sidecar directly and NOT call _load_midas
    depth2 = estimate_depth(img_bgr, smooth_sigma=0.0, img_path=img_path)
    assert midas_calls == 1  # Still 1 call!
    assert np.array_equal(depth1, depth2)

    # Third call after changing mtime (cache invalidation)
    import os
    stat = img_path.stat()
    os.utime(img_path, (stat.st_atime, stat.st_mtime + 5))

    depth3 = estimate_depth(img_bgr, smooth_sigma=0.0, img_path=img_path)
    assert midas_calls == 2  # Recomputed!


if __name__ == "__main__":
    try:
        test_render_parallax_frame_correctness()
        test_render_parallax_frame_axis_optimizations()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
