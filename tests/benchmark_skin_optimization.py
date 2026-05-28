import numpy as np
import time
import cv2

def _bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)

def _lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2BGR)
    return np.clip(bgr * 255.0, 0, 255).astype(np.uint8)

def benchmark():
    h, w = 1920, 1080
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[500:1500, 300:800] = True # ~25% of image
    target_lab = (70.0, 15.0, 20.0)
    deltas = (5.0, 2.0, -3.0)

    print(f"Benchmark: {w}x{h}, Mask coverage: {mask.mean()*100:.1f}%")

    # Simulation of the original O(N) logic
    start = time.time()
    for _ in range(10):
        lab = _bgr_to_lab(img)
        skin_pixels = lab[mask]
        src_means = skin_pixels.mean(axis=0)
        res_lab = lab.copy()
        res_lab[mask] = np.clip(skin_pixels + deltas, [0, -128, -128], [100, 127, 127])
        res_img = _lab_to_bgr(res_lab)
    old_time = (time.time() - start) / 10
    print(f"Original O(N) approach: {old_time:.4f}s")

    # Simulation of the optimized O(SkinPixels) logic
    start = time.time()
    for _ in range(10):
        skin_pixels_bgr = img[mask].reshape(1, -1, 3)
        skin_pixels_lab = _bgr_to_lab(skin_pixels_bgr).reshape(-1, 3)
        src_means = skin_pixels_lab.mean(axis=0)
        skin_pixels_lab += deltas
        np.clip(skin_pixels_lab, [0, -128, -128], [100, 127, 127], out=skin_pixels_lab)
        res_skin_bgr = _lab_to_bgr(skin_pixels_lab.reshape(1, -1, 3)).reshape(-1, 3)
        result = img.copy()
        result[mask] = res_skin_bgr
    new_time = (time.time() - start) / 10
    print(f"Optimized O(SkinPixels) approach: {new_time:.4f}s")

    speedup = old_time / new_time
    print(f"Speedup: {speedup:.2f}x")

    # Verify logical parity
    lab_old = _bgr_to_lab(img)
    lab_old[mask] += deltas
    val_old = _lab_to_bgr(lab_old)

    skin_bgr = img[mask].reshape(1, -1, 3)
    skin_lab = _bgr_to_lab(skin_bgr).reshape(-1, 3)
    skin_lab += deltas
    res_skin_bgr = _lab_to_bgr(skin_lab.reshape(1, -1, 3)).reshape(-1, 3)
    val_new = img.copy()
    val_new[mask] = res_skin_bgr

    diff = np.abs(val_old.astype(int) - val_new.astype(int)).max()
    print(f"Max pixel difference: {diff} (must be near 0)")

if __name__ == "__main__":
    benchmark()
