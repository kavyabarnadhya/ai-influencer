import time
import sys
from pathlib import Path
import json

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from comfyui_api import load_workflow, inject_workflow_values

def benchmark_loading():
    workflow_path = str(Path(__file__).parent.parent / "workflows" / "t2i_sdxl_upscale.json")

    # Cold start
    start_time = time.perf_counter()
    wf1 = load_workflow(workflow_path)
    cold_duration = time.perf_counter() - start_time

    # Warm start (cached)
    start_time = time.perf_counter()
    for _ in range(100):
        wf = load_workflow(workflow_path)
    warm_duration = (time.perf_counter() - start_time) / 100

    print(f"Workflow Loading Benchmark:")
    print(f"  Cold Load: {cold_duration*1000:.4f}ms")
    print(f"  Warm Load (cached + copy): {warm_duration*1000:.4f}ms")
    print(f"  Speedup: {cold_duration / warm_duration:.1f}x")

    # Verify immutability
    wf1["new_key"] = "leak"
    wf2 = load_workflow(workflow_path)
    if "new_key" in wf2:
        print("  FAILED: State leaked between loads!")
    else:
        print("  SUCCESS: No state leakage detected.")

def benchmark_injection():
    workflow_path = str(Path(__file__).parent.parent / "workflows" / "t2i_sdxl_upscale.json")
    wf = load_workflow(workflow_path)

    overrides = {
        "_claude_inject_prompt": {"inputs.text": "a beautiful landscape"},
        "_claude_inject_seed": {"inputs.seed": 12345},
    }

    # Benchmark injection
    start_time = time.perf_counter()
    iterations = 1000
    for _ in range(iterations):
        patched = inject_workflow_values(wf, overrides)
    duration = (time.perf_counter() - start_time) / iterations

    print(f"\nWorkflow Injection Benchmark ({iterations} iterations):")
    print(f"  Average injection time: {duration*1000:.4f}ms")

if __name__ == "__main__":
    benchmark_loading()
    benchmark_injection()
