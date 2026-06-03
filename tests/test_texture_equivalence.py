import numpy as np
import cv2
from pathlib import Path
from scripts.texture_integrity_check import compute_texture_score

def test_texture_equivalence(tmp_path):
    """
    Test that the optimized compute_texture_score returns the same results as
    the original (re-implemented here for comparison).
    """
    # Create a dummy image
    img_path = tmp_path / "test.png"
    h, w = 512, 512
    img_np = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    from PIL import Image
    Image.fromarray(img_np).save(img_path)

    # Re-implementation of the ORIGINAL slow algorithm for verification
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    expected_lap_var = float(lap.var())

    # Face crop
    y1, y2 = 0, int(h * 0.6)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    face_crop = gray[y1:y2, x1:x2]
    face_lap = cv2.Laplacian(face_crop, cv2.CV_64F)
    expected_face_var = float(face_lap.var())

    # FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    h2, w2 = magnitude.shape
    cy, cx = h2 // 2, w2 // 2
    r_inner = min(h2, w2) // 6
    mask = np.ones((h2, w2), dtype=bool)
    for i in range(h2):
        for j in range(w2):
            if (i - cy) ** 2 + (j - cx) ** 2 < r_inner ** 2:
                mask[i, j] = False
    expected_hf_ratio = float(magnitude[mask].sum() / (magnitude.sum() + 1e-8))

    # Run optimized version
    metrics = compute_texture_score(img_path)

    assert np.isclose(metrics["laplacian_var"], expected_lap_var)
    assert np.isclose(metrics["face_region_var"], expected_face_var)
    assert np.isclose(metrics["hf_ratio"], expected_hf_ratio)
    assert metrics["error"] is None
