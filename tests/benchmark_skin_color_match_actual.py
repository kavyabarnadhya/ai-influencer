import numpy as np
import time
import cv2
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from skin_color_match import _hsv_skin_filter, _apply_lab_delta

def benchmark():
    h, w = 1920, 1080
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    person_mask = np.zeros((h, w), dtype=np.uint8)
    # Simulate a person in the center-ish, 25% coverage (1200x500 = 600,000 pixels ~29%)
    person_mask[400:1600, 300:800] = 255
    target_lab = (70.0, 15.0, 20.0)

    print(f"Benchmarking skin_color_match on {w}x{h} image...")

    # Benchmark _hsv_skin_filter
    start = time.time()
    for _ in range(20):
        skin_mask = _hsv_skin_filter(img, person_mask)
    filter_time = (time.time() - start) / 20
    print(f"_hsv_skin_filter baseline: {filter_time:.4f}s")

    # We need a skin mask for the next benchmark
    skin_mask = _hsv_skin_filter(img, person_mask)

    # Benchmark _apply_lab_delta
    start = time.time()
    for _ in range(20):
        result = _apply_lab_delta(img, skin_mask, target_lab)
    delta_time = (time.time() - start) / 20
    print(f"_apply_lab_delta baseline: {delta_time:.4f}s")

    print(f"Total baseline time per frame: {filter_time + delta_time:.4f}s")

if __name__ == "__main__":
    benchmark()
