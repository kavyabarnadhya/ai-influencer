import sys
import time
from pathlib import Path
import concurrent.futures
import os

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from texture_integrity_check import compute_texture_score, SUPPORTED_EXTS

def benchmark_parallel():
    input_path = ROOT / "character" / "ananya" / "training_data"
    exts = tuple(e.lower() for e in SUPPORTED_EXTS)
    with os.scandir(input_path) as it:
        images = sorted([
            Path(entry.path) for entry in it
            if entry.is_file() and entry.name.lower().endswith(exts)
        ])

    print(f"Benchmarking with {len(images)} images...")

    # 1. Sequential
    start = time.perf_counter()
    seq_results = [compute_texture_score(p) for p in images]
    seq_time = time.perf_counter() - start
    print(f"Sequential: {seq_time:.4f}s")

    # 2. ThreadPoolExecutor
    for max_workers in [2, 4, 8]:
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            thread_results = list(executor.map(compute_texture_score, images))
        thread_time = time.perf_counter() - start
        print(f"ThreadPoolExecutor (workers={max_workers}): {thread_time:.4f}s (Speedup: {seq_time/thread_time:.2f}x)")

    # 3. ProcessPoolExecutor
    for max_workers in [2, 4, 8]:
        start = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            process_results = list(executor.map(compute_texture_score, images))
        process_time = time.perf_counter() - start
        print(f"ProcessPoolExecutor (workers={max_workers}): {process_time:.4f}s (Speedup: {seq_time/process_time:.2f}x)")

if __name__ == "__main__":
    benchmark_parallel()
