import cv2
import numpy as np
import time

def benchmark_in_place_cvt(h=1080, w=1080, iterations=100):
    # Use 1080x1080 to represent a typical ROI
    img = np.random.rand(h, w, 3).astype(np.float32)

    # Standard
    start = time.perf_counter()
    for _ in range(iterations):
        res1 = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    end = time.perf_counter()
    print(f"Standard cvtColor (float32): {(end-start)*1000/iterations:.4f} ms")

    # In-place
    img_copy = img.copy()
    start = time.perf_counter()
    for _ in range(iterations):
        cv2.cvtColor(img_copy, cv2.COLOR_BGR2Lab, dst=img_copy)
    end = time.perf_counter()
    print(f"In-place cvtColor (float32): {(end-start)*1000/iterations:.4f} ms")

    # Verify
    diff = np.abs(res1 - img_copy).max()
    print(f"Max diff: {diff}")

    # Test uint8 (for HSV filter)
    img_u8 = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    start = time.perf_counter()
    for _ in range(iterations):
        res_u8 = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)
    end = time.perf_counter()
    print(f"Standard cvtColor (uint8): {(end-start)*1000/iterations:.4f} ms")

    img_u8_copy = img_u8.copy()
    start = time.perf_counter()
    for _ in range(iterations):
        cv2.cvtColor(img_u8_copy, cv2.COLOR_BGR2HSV, dst=img_u8_copy)
    end = time.perf_counter()
    print(f"In-place cvtColor (uint8): {(end-start)*1000/iterations:.4f} ms")

    diff_u8 = np.abs(res_u8.astype(np.int16) - img_u8_copy.astype(np.int16)).max()
    print(f"Max diff uint8: {diff_u8}")

if __name__ == "__main__":
    benchmark_in_place_cvt()
