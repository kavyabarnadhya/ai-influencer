import numpy as np
import time
import cv2

# Constants from scripts/skin_color_match.py
_SKIN_HSV_LOWER1 = np.array([0, 30, 60], dtype=np.uint8)
_SKIN_HSV_UPPER1 = np.array([25, 200, 255], dtype=np.uint8)
_SKIN_HSV_LOWER2 = np.array([165, 30, 60], dtype=np.uint8)
_SKIN_HSV_UPPER2 = np.array([180, 200, 255], dtype=np.uint8)
_FACE_EXCLUSION_DILATE = 10

def _hsv_skin_filter_original(img_bgr: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, _SKIN_HSV_LOWER1, _SKIN_HSV_UPPER1)
    m2 = cv2.inRange(hsv, _SKIN_HSV_LOWER2, _SKIN_HSV_UPPER2)
    skin = cv2.bitwise_or(m1, m2).astype(bool)
    return skin & person_mask

def _hsv_skin_filter_roi(img_bgr: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    # Get bounding box of person_mask
    rows = np.any(person_mask, axis=1)
    cols = np.any(person_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return np.zeros_like(person_mask)

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    ymax += 1
    xmax += 1

    roi_bgr = img_bgr[ymin:ymax, xmin:xmax]
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(roi_hsv, _SKIN_HSV_LOWER1, _SKIN_HSV_UPPER1)
    m2 = cv2.inRange(roi_hsv, _SKIN_HSV_LOWER2, _SKIN_HSV_UPPER2)
    roi_skin = cv2.bitwise_or(m1, m2).astype(bool)

    full_skin = np.zeros(person_mask.shape, dtype=bool)
    full_skin[ymin:ymax, xmin:xmax] = roi_skin
    return full_skin & person_mask

def _face_exclusion_mask_original(img_bgr: np.ndarray, face_bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if face_bbox is None:
        return mask.astype(bool)
    x1, y1, x2, y2 = face_bbox
    mask[y1:y2, x1:x2] = 255
    if _FACE_EXCLUSION_DILATE > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _FACE_EXCLUSION_DILATE + 1, 2 * _FACE_EXCLUSION_DILATE + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(bool)

def _face_exclusion_mask_optimized(img_bgr: np.ndarray, face_bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if face_bbox is None:
        return mask.astype(bool)
    x1, y1, x2, y2 = face_bbox
    if _FACE_EXCLUSION_DILATE > 0:
        d = _FACE_EXCLUSION_DILATE
        ex1, ey1 = max(0, x1 - d), max(0, y1 - d)
        ex2, ey2 = min(w, x2 + d), min(h, y2 + d)
        roi_h, roi_w = ey2 - ey1, ex2 - ex1
        roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        rx1, ry1 = x1 - ex1, y1 - ey1
        rx2, ry2 = x2 - ex1, y2 - ey1
        roi_mask[ry1:ry2, rx1:rx2] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
        roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)
        mask[ey1:ey2, ex1:ex2] = roi_mask
    else:
        mask[y1:y2, x1:x2] = 255
    return mask.astype(bool)

def benchmark():
    h, w = 1920, 1080
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    # Realistic coverage for carousel slides (~20% area)
    person_mask = np.zeros((h, w), dtype=bool)
    person_mask[400:1500, 400:800] = True
    face_bbox = (400, 300, 600, 500)

    print(f"Benchmark: {w}x{h}, Person coverage: {person_mask.mean()*100:.1f}%")

    # Benchmark _hsv_skin_filter
    print("\nBenchmarking _hsv_skin_filter:")
    start = time.time()
    for _ in range(50):
        res_orig = _hsv_skin_filter_original(img, person_mask)
    orig_time = (time.time() - start) / 50
    print(f"  Original:  {orig_time:.4f}s")

    start = time.time()
    for _ in range(50):
        res_opt = _hsv_skin_filter_roi(img, person_mask)
    opt_time = (time.time() - start) / 50
    print(f"  Optimized: {opt_time:.4f}s")
    print(f"  Speedup:   {orig_time / opt_time:.2f}x")
    assert np.array_equal(res_orig, res_opt), "Logical mismatch in _hsv_skin_filter!"

    # Benchmark _face_exclusion_mask
    print("\nBenchmarking _face_exclusion_mask:")
    start = time.time()
    for _ in range(100):
        res_orig = _face_exclusion_mask_original(img, face_bbox)
    orig_time = (time.time() - start) / 100
    print(f"  Original:  {orig_time:.4f}s")

    start = time.time()
    for _ in range(100):
        res_opt = _face_exclusion_mask_optimized(img, face_bbox)
    opt_time = (time.time() - start) / 100
    print(f"  Optimized: {opt_time:.4f}s")
    print(f"  Speedup:   {orig_time / opt_time:.2f}x")
    assert np.array_equal(res_orig, res_opt), "Logical mismatch in _face_exclusion_mask!"

if __name__ == "__main__":
    benchmark()
