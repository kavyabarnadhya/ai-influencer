import sys
import time
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

# Add scripts to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import skin_color_match as scm

def benchmark_bolt_skin():
    # Mock YOLO to simulate hardware latency and count calls
    mock_yolo_instance = MagicMock()

    # We want to count how many times YOLO is CALLED
    yolo_call_count = 0

    def mocked_yolo_call(*args, **kwargs):
        nonlocal yolo_call_count
        yolo_call_count += 1
        time.sleep(0.05) # Simulate 50ms inference time

        # Return a mock result that looks like what the code expects
        mock_result = MagicMock()

        # Mocking the .cpu().numpy() chain
        mock_xyxy = MagicMock()
        mock_xyxy.cpu.return_value.numpy.return_value = np.array([[10, 10, 50, 50]])
        mock_result.boxes.xyxy = mock_xyxy

        mock_conf = MagicMock()
        mock_conf.argmax.return_value = 0
        mock_result.boxes.conf = mock_conf

        # Mocking results[0].masks.data[person_indices].any(dim=0).byte() * 255).cpu().numpy()
        mock_mask_data = MagicMock()
        mock_any = mock_mask_data.__getitem__.return_value.any.return_value
        mock_byte = mock_any.byte.return_value
        mock_mul = mock_byte.__mul__.return_value
        mock_cpu = mock_mul.cpu.return_value
        # Mocking mask data to be the same size as the input image
        mock_cpu.numpy.return_value = np.zeros((1920, 1080), dtype=np.uint8)
        mock_result.masks.data = mock_mask_data

        # Mocking classes as a torch-like tensor
        mock_cls = MagicMock()
        mock_cls.__eq__.return_value.nonzero.return_value = (np.array([0]),)
        mock_result.boxes.cls = mock_cls

        return [mock_result]

    mock_yolo_instance.side_effect = mocked_yolo_call

    # Patch the model loaders to return our mock
    scm._load_face_model = MagicMock(return_value=mock_yolo_instance)
    scm._load_seg_model = MagicMock(return_value=mock_yolo_instance)

    # Dummy inputs: 1080p
    h, w = 1920, 1080
    dummy_img = np.zeros((h, w, 3), dtype=np.uint8)
    dummy_img[400:1600, 300:800] = [100, 120, 150] # Some "skin"

    # Create a dummy face_ref file
    face_ref_path = ROOT / "tests" / "dummy_face_ref.png"
    import cv2
    cv2.imwrite(str(face_ref_path), dummy_img)

    print(f"--- Starting Baseline Benchmark ---")

    # Warm up caches
    scm._sample_face_skin_lab_cached.cache_clear()

    start_time = time.time()
    # match_body_skin_to_face_ref(slide_path, face_ref_path, out_path, img_bgr)
    # We pass img_bgr to skip disk read for the slide
    scm.match_body_skin_to_face_ref(None, face_ref_path, None, img_bgr=dummy_img)
    end_time = time.time()

    duration = end_time - start_time
    print(f"Total duration: {duration:.4f}s")
    print(f"YOLO calls: {yolo_call_count}")

    # Clean up
    if face_ref_path.exists():
        face_ref_path.unlink()

if __name__ == "__main__":
    benchmark_bolt_skin()
