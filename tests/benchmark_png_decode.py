import time
import cv2
import os
import numpy as np

def benchmark_io():
    # Create a dummy 1080p image (typical carousel size)
    img = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    cv2.imwrite("dummy.png", img)

    start = time.time()
    for _ in range(100):
        # MediaPipe and YOLO both decode the image.
        # We want to measure the cost of one extra decode.
        _ = cv2.imread("dummy.png")
    end = time.time()

    avg_ms = (end - start) * 1000 / 100
    print(f"Time to read 100 1080p PNGs: {end - start:.4f}s (Avg: {avg_ms:.2f}ms/img)")

    if os.path.exists("dummy.png"):
        os.remove("dummy.png")

if __name__ == "__main__":
    benchmark_io()
