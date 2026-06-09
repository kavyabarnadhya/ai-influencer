import time
import numpy as np
import cv2
import os

def benchmark_png_compression():
    size = (1080, 1920)
    # Random noise is hard to compress, let's use something more representative
    # img_np = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)

    # Gradient image
    img_np = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for i in range(size[1]):
        img_np[i, :, 0] = i % 256
        img_np[i, :, 1] = (i // 2) % 256
        img_np[i, :, 2] = (i // 3) % 256

    print(f"Benchmarking PNG compression for {size}...")

    for level in [0, 1, 3, 6, 9]:
        start = time.time()
        cv2.imwrite(f"test_l{level}.png", img_np, [cv2.IMWRITE_PNG_COMPRESSION, level])
        t = time.time() - start
        sz = os.path.getsize(f"test_l{level}.png") / 1024 / 1024
        print(f"Level {level}: {t:.4f}s, {sz:.2f}MB")
        os.remove(f"test_l{level}.png")

if __name__ == "__main__":
    benchmark_png_compression()
