import cv2
import numpy as np
import time

def test_add_weighted_slice():
    img = np.zeros((1920, 1080, 3), dtype=np.float32)
    mask = np.random.rand(1920, 1080).astype(np.float32)

    print("Testing cv2.addWeighted on slice...")
    try:
        # dst = src1*alpha + src2*beta + gamma
        # We want: img[..., 0] = img[..., 0] + mask * dL
        # So: src1=img[..., 0], alpha=1.0, src2=mask, beta=0.5, gamma=0.0
        cv2.addWeighted(img[..., 0], 1.0, mask, 0.5, 0.0, dst=img[..., 0])
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")

def benchmark_ops():
    h, w = 1920, 1080
    img = np.random.rand(h, w, 3).astype(np.float32)
    mask = np.random.rand(h, w).astype(np.float32)
    dL = 10.0

    # NumPy
    start = time.perf_counter()
    img[..., 0] += mask * dL
    end = time.perf_counter()
    print(f"NumPy += mask * dL: {(end-start)*1000:.2f} ms")

    # cv2.scaleAdd (if it works on slices)
    img = np.random.rand(h, w, 3).astype(np.float32)
    try:
        start = time.perf_counter()
        # dst = src1 * scale + src2
        # src1=mask, scale=dL, src2=img[..., 0]
        cv2.scaleAdd(mask, dL, img[..., 0], dst=img[..., 0])
        end = time.perf_counter()
        print(f"cv2.scaleAdd: {(end-start)*1000:.2f} ms")
    except Exception as e:
        print(f"cv2.scaleAdd failed: {e}")

if __name__ == "__main__":
    test_add_weighted_slice()
    benchmark_ops()
