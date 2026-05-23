import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from comfyui_api import inject_workflow_values

def create_large_workflow(num_nodes: int = 10000) -> dict:
    workflow = {}
    for i in range(num_nodes):
        node_id = str(i)
        title = f"Node_{i % 10}"  # 10 unique titles
        workflow[node_id] = {
            "inputs": {
                "text": f"original text {i}",
                "seed": i,
                "nested": {
                    "field": "val"
                }
            },
            "_meta": {"title": title}
        }
    return workflow

def benchmark():
    num_nodes = 10000
    iterations = 2000
    workflow = create_large_workflow(num_nodes)

    # Overrides that match many nodes (1000 nodes each)
    overrides = {
        "Node_0": {"inputs.text": "new text", "inputs.seed": 999},
        "Node_1": {"inputs.nested.field": "new val"},
        "Node_2": {"inputs.text": "same text"},
    }

    # First call to warm up title cache
    workflow = inject_workflow_values(workflow, {})

    # Benchmark redundant patches (values already match)
    workflow_patched = inject_workflow_values(workflow, overrides)

    print(f"Benchmarking with {num_nodes} nodes, {iterations} iterations...")

    start_time = time.perf_counter()
    for _ in range(iterations):
        inject_workflow_values(workflow_patched, overrides)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = (total_time / iterations) * 1000  # ms

    print(f"Average time per call (redundant): {avg_time:.4f} ms")

    # Benchmark non-redundant patches
    start_time = time.perf_counter()
    for i in range(iterations):
        dynamic_overrides = {
            "Node_0": {"inputs.seed": i},
        }
        inject_workflow_values(workflow, dynamic_overrides)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = (total_time / iterations) * 1000  # ms
    print(f"Average time per call (dynamic): {avg_time:.4f} ms")

if __name__ == "__main__":
    benchmark()
