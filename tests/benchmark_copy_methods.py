import time
import json
import sys
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Mocking _scan_workflow_titles to avoid dependencies for this micro-benchmark
def mock_scan_workflow_titles(wf):
    return {}

def benchmark_copy_methods(iterations=1000):
    wf_path = ROOT / "workflows" / "flux_schnell.json"
    with open(wf_path, "r") as f:
        original = json.load(f)

    # 1. Deepcopy
    start = time.perf_counter()
    for _ in range(iterations):
        _ = copy.deepcopy(original)
    end = time.perf_counter()
    print(f"deepcopy: {(end - start)*1000/iterations:.4f} ms")

    # 2. JSON loads/dumps
    start = time.perf_counter()
    json_str = json.dumps(original)
    for _ in range(iterations):
        _ = json.loads(json_str)
    end = time.perf_counter()
    print(f"json loads: {(end - start)*1000/iterations:.4f} ms")

    # 3. Shallow copy
    start = time.perf_counter()
    for _ in range(iterations):
        _ = original.copy()
    end = time.perf_counter()
    print(f"shallow copy: {(end - start)*1000/iterations:.4f} ms")

if __name__ == "__main__":
    benchmark_copy_methods()
