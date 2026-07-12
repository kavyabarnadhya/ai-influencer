import time
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from comfyui_api import load_workflow

def benchmark_load_workflow(iterations=100):
    wf_path = str(ROOT / "workflows" / "flux_schnell.json")

    # Warm up
    load_workflow(wf_path)

    start = time.perf_counter()
    for _ in range(iterations):
        load_workflow(wf_path)
    end = time.perf_counter()

    avg_time = (end - start) / iterations
    print(f"Average time to load_workflow: {avg_time*1000:.4f} ms")

if __name__ == "__main__":
    benchmark_load_workflow()
