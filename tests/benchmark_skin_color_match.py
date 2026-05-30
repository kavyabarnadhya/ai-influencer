"""
Benchmark + equivalence guard for the ROI optimizations in scripts/skin_color_match.py.

The OPTIMIZED functions are imported from the real module so this test actually
guards the shipped code (no copy-paste drift). The naive reference implementations
below are the correctness oracle; each optimized function must produce a
bit-identical mask on every test case, including faces near the image border.

Run: python tests/benchmark_skin_color_match.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Import the REAL optimized functions + constants under test.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.skin_color_match import (  # noqa: E402
    _FACE_EXCLUSION_DILATE,
    _SKIN_HSV_LOWER1,
    _SKIN_HSV_LOWER2,
    _SKIN_HSV_UPPER1,
    _SKIN_HSV_UPPER2,
    _face_exclusion_mask,
    _hsv_skin_filter,
)


# --- naive reference implementations (correctness oracle) ---

def _hsv_skin_filter_naive(img_bgr: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, _SKIN_HSV_LOWER1, _SKIN_HSV_UPPER1)
    m2 = cv2.inRange(hsv, _SKIN_HSV_LOWER2, _SKIN_HSV_UPPER2)
    skin = cv2.bitwise_or(m1, m2).astype(bool)
    return skin & person_mask


def _face_exclusion_mask_naive(img_bgr: np.ndarray, face_bbox) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if face_bbox is None:
        return mask.astype(bool)
    x1, y1, x2, y2 = face_bbox
    mask[y1:y2, x1:x2] = 255
    if _FACE_EXCLUSION_DILATE > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * _FACE_EXCLUSION_DILATE + 1, 2 * _FACE_EXCLUSION_DILATE + 1),
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(bool)


# --- equivalence: optimized (real) must match naive on every case ---

def test_equivalence():
    h, w = 1920, 1080
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    # _hsv_skin_filter: mid-image person mask + person mask touching all edges
    masks = []
    m_mid = np.zeros((h, w), dtype=bool)
    m_mid[400:1500, 400:800] = True
    masks.append(("mid", m_mid))
    m_edge = np.zeros((h, w), dtype=bool)
    m_edge[0:h, 0:50] = True          # left edge column
    m_edge[h - 50:h, 0:w] = True      # bottom edge row
    masks.append(("edge", m_edge))
    for label, pm in masks:
        assert np.array_equal(
            _hsv_skin_filter(img, pm), _hsv_skin_filter_naive(img, pm)
        ), f"_hsv_skin_filter mismatch on {label} mask"

    # _face_exclusion_mask: mid + corner + near-edge (exercises the ROI border-clamp paths)
    bboxes = [
        ("mid", (400, 300, 600, 500)),
        ("corner_tl", (0, 0, 150, 150)),
        ("near_edge_br", (w - 120, h - 120, w - 5, h - 5)),
    ]
    for label, bb in bboxes:
        assert np.array_equal(
            _face_exclusion_mask(img, bb), _face_exclusion_mask_naive(img, bb)
        ), f"_face_exclusion_mask mismatch on {label} bbox"

    assert np.array_equal(
        _face_exclusion_mask(img, None), _face_exclusion_mask_naive(img, None)
    ), "_face_exclusion_mask mismatch on None bbox"

    print("Equivalence: PASS (hsv mid/edge, face mid/corner/edge/None)")


def benchmark():
    h, w = 1920, 1080
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    person_mask = np.zeros((h, w), dtype=bool)
    person_mask[400:1500, 400:800] = True  # ~20% coverage
    face_bbox = (400, 300, 600, 500)

    print(f"Benchmark: {w}x{h}, person coverage {person_mask.mean() * 100:.1f}%")

    def _time(fn, args, n):
        start = time.time()
        for _ in range(n):
            fn(*args)
        return (time.time() - start) / n

    print("\n_hsv_skin_filter:")
    o = _time(_hsv_skin_filter_naive, (img, person_mask), 50)
    p = _time(_hsv_skin_filter, (img, person_mask), 50)
    print(f"  naive {o:.4f}s | optimized {p:.4f}s | speedup {o / p:.2f}x")

    print("\n_face_exclusion_mask:")
    o = _time(_face_exclusion_mask_naive, (img, face_bbox), 100)
    p = _time(_face_exclusion_mask, (img, face_bbox), 100)
    print(f"  naive {o:.4f}s | optimized {p:.4f}s | speedup {o / p:.2f}x")


if __name__ == "__main__":
    test_equivalence()
    benchmark()
