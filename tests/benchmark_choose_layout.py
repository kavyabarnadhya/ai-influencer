import os
import time
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_training_data as prep

def benchmark_choose_layout(num_files=100, iterations=1000):
    print(f"Benchmarking choose_layout with {num_files} files in the directory over {iterations} iterations...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Populate directory with files.
        # Ensure a supported image is present so the early-exit check matches quickly,
        # but place it after some other files to be realistic.
        (tmp_path / "notes.txt").write_text("AnanyaAI caption notes")
        for i in range(num_files - 1):
            if i == 50:
                # Place a supported image in the middle
                (tmp_path / f"image_canonical_{i:04d}.png").touch()
            else:
                (tmp_path / f"text_caption_{i:04d}.txt").write_text("text content")

        char_cfg = {
            "seeds_dir": "character/ananya/seeds",
            "training_data_dir": str(tmp_path)
        }

        # Old unoptimized choose_layout simulation using get_flux_images (sort/list/allocate)
        def old_get_flux_images(training_dir):
            if not training_dir.exists():
                return []
            exts = tuple(e.lower() for e in prep.SUPPORTED_IMAGE_EXTENSIONS)
            with os.scandir(training_dir) as it:
                return sorted(
                    Path(entry.path)
                    for entry in it
                    if entry.is_file() and entry.name.lower().endswith(exts)
                )

        def old_choose_layout(char_cfg):
            training_dir = prep.get_training_data_dir(char_cfg)
            return "flux" if old_get_flux_images(training_dir) else "sdxl"

        # Warm up both functions
        old_choose_layout(char_cfg)
        prep.choose_layout(char_cfg, "auto")

        # Measure baseline (old get_flux_images check)
        start_old = time.perf_counter()
        for _ in range(iterations):
            old_choose_layout(char_cfg)
        end_old = time.perf_counter()
        old_time = (end_old - start_old) / iterations

        # Measure optimized (new has_flux_images check)
        start_new = time.perf_counter()
        for _ in range(iterations):
            prep.choose_layout(char_cfg, "auto")
        end_new = time.perf_counter()
        new_time = (end_new - start_new) / iterations

        print("\n" + "="*50)
        print(f"Old baseline choose_layout: {old_time*1000:.6f} ms")
        print(f"New optimized choose_layout: {new_time*1000:.6f} ms")
        print(f"Speedup: {old_time / new_time:.2f}x")
        print("="*50 + "\n")

if __name__ == "__main__":
    benchmark_choose_layout()
