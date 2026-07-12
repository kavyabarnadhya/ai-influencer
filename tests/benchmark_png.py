import cv2
import numpy as np
import time
import os

def benchmark_png_compression(h=1920, w=1080, iterations=5):
    # Create a dummy photographic-style image (noise + gradients)
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    # Add some structure
    cv2.GaussianBlur(img, (51, 51), 0, dst=img)

    results = {}
    for level in [1, 3, 6, 9]:
        start = time.perf_counter()
        for _ in range(iterations):
            _, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, level])
        end = time.perf_counter()

        avg_time = (end - start) / iterations
        size_kb = len(buf) / 1024
        results[level] = (avg_time, size_kb)
        print(f"Level {level}: {avg_time*1000:.2f} ms, {size_kb:.2f} KB")

if __name__ == "__main__":
    benchmark_png_compression()
