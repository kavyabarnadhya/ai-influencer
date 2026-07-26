import os
import time
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

def benchmark_sdxl_discovery(num_images=100, iterations=100):
    print(f"--- SDXL Discovery Benchmark ({num_images} images/captions, {iterations} iterations) ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create images and corresponding captions
        for i in range(num_images):
            (tmp_path / f"img_{i:04d}.png").touch()
            if i % 10 != 0:  # 90% captions
                (tmp_path / f"img_{i:04d}.txt").touch()

        # 1. Baseline logic (Old unoptimized dual-pass approach)
        def old_discovery_approach(mode_dir):
            # Pass 1: get images (calling get_seed_images style)
            images = []
            if mode_dir.exists():
                with os.scandir(mode_dir) as it:
                    images = sorted([
                        Path(entry.path)
                        for entry in it
                        if entry.is_file() and entry.name.lower().endswith(".png")
                    ])

            # Pass 2: get existing captions
            existing_captions = set()
            if mode_dir.exists():
                with os.scandir(mode_dir) as it:
                    for entry in it:
                        if entry.is_file() and entry.name.lower().endswith(".txt"):
                            existing_captions.add(entry.name)

            captions = []
            for img in images:
                cap_name = img.name.rsplit(".", 1)[0] + ".txt"
                if cap_name in existing_captions:
                    captions.append(mode_dir / cap_name)
            return images, captions

        # Warm up
        _, _ = old_discovery_approach(tmp_path)

        start = time.perf_counter()
        for _ in range(iterations):
            _ = old_discovery_approach(tmp_path)
        end = time.perf_counter()
        old_time = (end - start) / iterations

        # 2. Optimized logic (Single-pass consolidated approach)
        def optimized_discovery_approach(mode_dir):
            images = []
            existing_captions = set()
            if mode_dir.exists():
                with os.scandir(mode_dir) as it:
                    for entry in it:
                        if entry.is_file():
                            name_low = entry.name.lower()
                            if name_low.endswith(".png"):
                                images.append(Path(entry.path))
                            elif name_low.endswith(".txt"):
                                existing_captions.add(entry.name)
            images.sort()

            captions = []
            for img in images:
                cap_name = img.name.rsplit(".", 1)[0] + ".txt"
                if cap_name in existing_captions:
                    captions.append(mode_dir / cap_name)
            return images, captions

        # Warm up
        _, _ = optimized_discovery_approach(tmp_path)

        start = time.perf_counter()
        for _ in range(iterations):
            _ = optimized_discovery_approach(tmp_path)
        end = time.perf_counter()
        optimized_time = (end - start) / iterations

        print(f"  Old Dual-Pass logic avg: {old_time*1000:.4f} ms")
        print(f"  New Consolidated Single-Pass logic avg: {optimized_time*1000:.4f} ms")
        print(f"  Speedup: {old_time / optimized_time:.2f}x")
        print(f"  Saves {(old_time - optimized_time)*1000:.4f} ms per directory scan!")
        print("="*60 + "\n")

if __name__ == "__main__":
    benchmark_sdxl_discovery()
