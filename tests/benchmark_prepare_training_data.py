import os
import time
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_training_data as prep

def benchmark_validate_flux(num_images=100, iterations=100):
    print(f"Benchmarking validate_flux logic with {num_images} images/captions (console.print disabled)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Create supported images, corresponding txt captions, some unsupported files, and a few duplicates
        for i in range(num_images):
            (tmp_path / f"img_{i:04d}.png").touch()
            # 90% have captions
            if i % 10 != 0:
                (tmp_path / f"img_{i:04d}.txt").write_text("AnanyaAI is in Mumbai posing naturally", encoding="utf-8")

        # Add some unsupported images
        for i in range(10):
            (tmp_path / f"unsupported_{i:04d}.tif").touch()

        # Mock the trigger word and target directory
        char_cfg = {
            "trigger_word": "AnanyaAI",
            "seeds_dir": "character/ananya/seeds",
            "training_data_dir": str(tmp_path)
        }
        cfg = {}

        # Temporarily mock console.print to a no-op
        original_print = prep.console.print
        prep.console.print = lambda *args, **kwargs: None

        # Warm up
        prep.validate_flux(cfg, char_cfg)

        # 2. Benchmark the Optimized version (Single-pass os.scandir + O(1) set lookup)
        start = time.perf_counter()
        for _ in range(iterations):
            prep.validate_flux(cfg, char_cfg)
        end = time.perf_counter()
        optimized_time = (end - start) / iterations

        # 3. Simulate the Old (Unoptimized) logic on the exact same temp directory
        # Let's recreate the old logic exactly to compare wall-clock timings.
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

        def old_get_unsupported_images(training_dir):
            if not training_dir.exists():
                return []
            exts = tuple(e.lower() for e in prep.UNSUPPORTED_IMAGE_EXTENSIONS)
            with os.scandir(training_dir) as it:
                return sorted(
                    Path(entry.path)
                    for entry in it
                    if entry.is_file() and entry.name.lower().endswith(exts)
                )

        def old_validate_flux_simulation():
            images = old_get_flux_images(tmp_path)
            unsupported = old_get_unsupported_images(tmp_path)

            # 100 stat calls for duplicate keys
            _ = [
                prep.normalize_duplicate_key(path) for path in images
            ]

            # exists check for each image in missing captions
            missing_captions = [img.name for img in images if not img.with_suffix(".txt").exists()]

            for img in images:
                cap_path = img.with_suffix(".txt")
                # exists check in the loop
                if not cap_path.exists():
                    continue
                try:
                    caption = cap_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                # search terms
                _ = "AnanyaAI" in caption
                _ = prep.find_forbidden_caption_terms(caption)

        start = time.perf_counter()
        for _ in range(iterations):
            old_validate_flux_simulation()
        end = time.perf_counter()
        old_time = (end - start) / iterations

        # Restore original console.print
        prep.console.print = original_print

        print("\n" + "="*50)
        print(f"Old baseline logic avg: {old_time*1000:.4f} ms")
        print(f"New optimized logic avg: {optimized_time*1000:.4f} ms")
        print(f"Speedup: {old_time / optimized_time:.2f}x")
        print("="*50 + "\n")

if __name__ == "__main__":
    benchmark_validate_flux()
