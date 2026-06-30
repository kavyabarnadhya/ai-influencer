import os
import time
from pathlib import Path
import tempfile

def benchmark_discovery():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create 100 images and 900 dummy files
        for i in range(100):
            (tmp_path / f"img_{i}.png").touch()
        for i in range(900):
            (tmp_path / f"file_{i}.txt").touch()

        SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
        exts = tuple(e.lower() for e in SUPPORTED_EXTS)

        # Path.iterdir() benchmark
        start = time.perf_counter()
        for _ in range(100):
            images_iterdir = sorted([p for p in tmp_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
        iterdir_time = (time.perf_counter() - start) / 100
        print(f"Path.iterdir() (10% hit) avg time: {iterdir_time*1000:.4f}ms")

        # os.scandir() benchmark (returning Path objects)
        start = time.perf_counter()
        for _ in range(100):
            with os.scandir(tmp_path) as it:
                images_scandir = sorted([
                    Path(entry.path) for entry in it
                    if entry.is_file() and entry.name.lower().endswith(exts)
                ])
        scandir_path_time = (time.perf_counter() - start) / 100
        print(f"os.scandir() (Path objects, 10% hit) avg time: {scandir_path_time*1000:.4f}ms")

        # os.scandir() benchmark (returning strings)
        start = time.perf_counter()
        for _ in range(100):
            with os.scandir(tmp_path) as it:
                images_scandir_str = sorted([
                    entry.path for entry in it
                    if entry.is_file() and entry.name.lower().endswith(exts)
                ])
        scandir_str_time = (time.perf_counter() - start) / 100
        print(f"os.scandir() (strings, 10% hit) avg time: {scandir_str_time*1000:.4f}ms")

        print(f"Speedup (scandir Path vs iterdir): {iterdir_time / scandir_path_time:.2f}x")
        print(f"Speedup (scandir str vs iterdir): {iterdir_time / scandir_str_time:.2f}x")

if __name__ == "__main__":
    benchmark_discovery()
