import time
import numpy as np
import cv2
from PIL import Image
import os

def benchmark_resize():
    # Simulate a high-res AI output (e.g. 1536x1536 or similar)
    size = (1536, 1536)
    target_size = (1080, 1920)
    img_np = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    img_pil = Image.fromarray(img_np)

    print(f"Benchmarking resize from {size} to {target_size}...")

    # Pillow resize
    start = time.time()
    for _ in range(10):
        res_pil = img_pil.resize(target_size, Image.LANCZOS)
    pil_time = (time.time() - start) / 10
    print(f"Pillow (LANCZOS) average: {pil_time:.4f}s")

    # OpenCV resize
    start = time.time()
    for _ in range(10):
        res_cv = cv2.resize(img_np, target_size, interpolation=cv2.INTER_LANCZOS4)
    cv_time = (time.time() - start) / 10
    print(f"OpenCV (LANCZOS4) average: {cv_time:.4f}s")
    print(f"Resize speedup: {pil_time/cv_time:.2f}x")

def benchmark_io():
    size = (1080, 1920)
    img_np = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    test_file = "test_io.png"

    print(f"\nBenchmarking I/O for {size} PNG...")

    start = time.time()
    for _ in range(5):
        cv2.imwrite(test_file, img_np)
        _ = cv2.imread(test_file)
    io_time = (time.time() - start) / 5
    print(f"OpenCV Write+Read cycle average: {io_time:.4f}s")

    if os.path.exists(test_file):
        os.remove(test_file)

if __name__ == "__main__":
    benchmark_resize()
    benchmark_io()
