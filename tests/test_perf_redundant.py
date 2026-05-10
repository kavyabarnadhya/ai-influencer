import time
import sys
from pathlib import Path
from typing import Any

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from comfyui_api import load_workflow, inject_workflow_values

def benchmark():
    workflow_path = str(Path(__file__).parent.parent / "workflows" / "t2i_sdxl_upscale.json")
    wf = load_workflow(workflow_path)

    base_overrides = {
        "_claude_inject_prompt": {"inputs.text": "a beautiful landscape"},
        "_claude_inject_latent": {"inputs.width": 1024, "inputs.height": 1024},
    }
    wf_patched = inject_workflow_values(wf, base_overrides)

    # 1. All redundant
    all_redundant = {
        "_claude_inject_prompt": {"inputs.text": "a beautiful landscape"},
        "_claude_inject_latent": {"inputs.width": 1024, "inputs.height": 1024},
    }

    # 2. Mixed (one redundant, one new)
    mixed = {
        "_claude_inject_prompt": {"inputs.text": "a beautiful landscape"},
        "_claude_inject_seed": {"inputs.seed": 999},
    }

    # 3. All new
    all_new = {
        "_claude_inject_seed": {"inputs.seed": 888},
        "_claude_inject_checkpoint": {"inputs.ckpt_name": "other.safetensors"},
    }

    iterations = 10000

    print(f"Running {iterations} iterations...")

    start = time.perf_counter()
    for _ in range(iterations):
        inject_workflow_values(wf_patched, all_redundant)
    t_all_red = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        inject_workflow_values(wf_patched, mixed)
    t_mixed = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        inject_workflow_values(wf_patched, all_new)
    t_all_new = time.perf_counter() - start

    print(f"All Redundant: {t_all_red*1000:.2f}ms")
    print(f"Mixed:         {t_mixed*1000:.2f}ms")
    print(f"All New:       {t_all_new*1000:.2f}ms")

if __name__ == "__main__":
    benchmark()
