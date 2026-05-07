import time
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from comfyui_api import load_workflow, inject_workflow_values

def benchmark_redundant_overrides():
    workflow_path = str(Path(__file__).parent.parent / "workflows" / "t2i_sdxl_upscale.json")
    wf = load_workflow(workflow_path)

    # Simulate batch_generate base overrides
    base_overrides = {
        "_claude_inject_prompt": {"inputs.text": "a beautiful landscape"},
        "_claude_inject_negative": {"inputs.text": "blurry, low quality"},
        "_claude_inject_latent": {"inputs.width": 1024, "inputs.height": 1024},
        "_claude_inject_checkpoint": {"inputs.ckpt_name": "sd_xl_base_1.0.safetensors"},
        "_claude_inject_upscaler": {"inputs.model_name": "4x-UltraSharp.pth"},
    }
    wf_patched = inject_workflow_values(wf, base_overrides)

    # Redundant overrides as currently in batch_generate
    redundant_overrides = {
        "_claude_inject_seed": {"inputs.seed": 12345},
        "_claude_inject_upscaler": {"inputs.model_name": "4x-UltraSharp.pth"},
    }

    # Clean overrides
    clean_overrides = {
        "_claude_inject_seed": {"inputs.seed": 12345},
    }

    iterations = 5000

    start = time.perf_counter()
    for _ in range(iterations):
        inject_workflow_values(wf_patched, redundant_overrides)
    redundant_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        inject_workflow_values(wf_patched, clean_overrides)
    clean_time = time.perf_counter() - start

    print(f"Redundant time: {redundant_time*1000:.2f}ms")
    print(f"Clean time: {clean_time*1000:.2f}ms")
    print(f"Improvement: {(redundant_time - clean_time) / redundant_time * 100:.1f}%")

if __name__ == "__main__":
    benchmark_redundant_overrides()
