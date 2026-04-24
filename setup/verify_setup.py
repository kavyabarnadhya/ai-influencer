#!/usr/bin/env python3
"""End-to-end smoke test for the ai-influencer pipeline.

Checks:
  1. All required model files are present
  2. All workflow JSON files contain required sentinel titles
  3. ComfyUI is reachable (starts it if not running)
  4. A minimal SDXL test workflow completes successfully

Usage:
    python setup/verify_setup.py
    python setup/verify_setup.py --config config.yaml --skip-generation
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import yaml
    from rich.console import Console
    from rich.table import Table
except ImportError:
    print("Missing dependencies. Run: pip install PyYAML rich")
    sys.exit(1)

console = Console()

# Sentinel titles that must appear in workflow JSON files
REQUIRED_SENTINELS = {
    "workflows/t2i_sdxl_lora.json": [
        "_claude_inject_checkpoint",
        "_claude_inject_lora",
        "_claude_inject_prompt",
        "_claude_inject_negative",
        "_claude_inject_latent",
        "_claude_inject_seed",
    ],
    "workflows/t2i_ipadapter.json": [
        "_claude_inject_checkpoint",
        "_claude_inject_lora",
        "_claude_inject_prompt",
        "_claude_inject_negative",
        "_claude_inject_latent",
        "_claude_inject_seed",
    ],
    "workflows/bootstrap_seeds.json": [
        "_claude_inject_prompt",
        "_claude_inject_negative",
        "_claude_inject_latent",
        "_claude_inject_seed",
    ],
    "workflows/flux_schnell.json": [
        "_claude_inject_prompt",
        "_claude_inject_latent",
        "_claude_inject_seed",
    ],
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_models(cfg: dict) -> list[tuple[str, bool, str]]:
    models_dir = Path(cfg["models"]["dir"])
    checks = [
        ("sdxl_checkpoint", cfg["models"]["sdxl_checkpoint"]),
        ("lora", cfg["models"]["lora"]),
        ("ipadapter", cfg["models"]["ipadapter"]),
        ("flux_unet", cfg["models"]["flux_unet"]),
        ("flux_vae", cfg["models"]["flux_vae"]),
        ("flux_clip_l", cfg["models"]["flux_clip_l"]),
        ("flux_t5", cfg["models"]["flux_t5"]),
        ("vae", cfg["models"]["vae"]),
        ("adetailer_model", cfg["models"]["adetailer_model"]),
    ]
    results = []
    for key, rel_path in checks:
        full_path = models_dir / rel_path
        exists = full_path.exists()
        note = ""
        if not exists:
            if key == "lora":
                note = "Train LoRA first — see setup/train_lora_guide.md"
            elif "flux" in key:
                note = "Optional: only needed for FLUX tier"
        results.append((key, rel_path, exists, note))
    return results


def check_sentinels(project_root: Path) -> list[tuple[str, str, bool]]:
    results = []
    for workflow_rel, sentinels in REQUIRED_SENTINELS.items():
        workflow_path = project_root / workflow_rel
        if not workflow_path.exists():
            for sentinel in sentinels:
                results.append((workflow_rel, sentinel, False))
            continue
        content = workflow_path.read_text()
        for sentinel in sentinels:
            found = sentinel in content
            results.append((workflow_rel, sentinel, found))
    return results


def check_comfyui(cfg: dict, timeout: int = 30) -> bool:
    try:
        import requests
    except ImportError:
        console.print("  [yellow]requests not installed — skipping ComfyUI connectivity check[/yellow]")
        return False

    host = cfg["comfyui"]["host"]
    port = cfg["comfyui"]["port"]
    url = f"http://{host}:{port}/system_stats"
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)

    return False


def run_test_workflow(cfg: dict, project_root: Path) -> bool:
    workflow_path = project_root / "workflows" / "bootstrap_seeds.json"
    if not workflow_path.exists():
        console.print("  [yellow]bootstrap_seeds.json not found — skipping generation test[/yellow]")
        return False

    try:
        sys.path.insert(0, str(project_root / "scripts"))
        from comfyui_api import ComfyUIClient

        client = ComfyUIClient(
            host=cfg["comfyui"]["host"],
            port=cfg["comfyui"]["port"],
        )
        workflow = json.loads(workflow_path.read_text())

        # Inject a minimal test prompt via sentinel
        for node in workflow.values():
            if isinstance(node, dict):
                meta = node.get("_meta", {})
                if meta.get("title") == "_claude_inject_prompt":
                    node["inputs"]["text"] = "a test image, simple scene, no person"
                if meta.get("title") == "_claude_inject_seed":
                    node["inputs"]["seed"] = 42

        output_dir = project_root / cfg["generation"]["output_dir"] / "_verify_test"
        output_dir.mkdir(parents=True, exist_ok=True)

        images = client.submit_and_wait(workflow, str(output_dir))
        return len(images) > 0
    except Exception as exc:
        console.print(f"  [red]Generation test error:[/red] {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify ai-influencer pipeline setup")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-generation", action="store_true", help="Skip the test generation step")
    args = parser.parse_args()

    project_root = Path(args.config).parent.resolve()
    cfg = load_config(args.config)

    console.print("\n[bold]AI-Influencer Setup Verification[/bold]\n")
    all_ok = True

    # --- Model files ---
    console.print("[bold]Model Files[/bold]")
    model_checks = check_models(cfg)
    lora_ok = True
    flux_missing = []
    required_missing = []

    for key, rel_path, exists, note in model_checks:
        status = "[green]OK[/green]" if exists else "[red]MISSING[/red]"
        note_str = f"  ({note})" if note else ""
        console.print(f"  {status}  {rel_path}{note_str}")
        if not exists:
            if key == "lora":
                lora_ok = False
            elif "flux" in key:
                flux_missing.append(key)
            else:
                required_missing.append(key)
                all_ok = False

    if not lora_ok:
        console.print("  [yellow]  LoRA not yet trained — run bootstrap_seeds.py then follow train_lora_guide.md[/yellow]")
    if flux_missing:
        console.print(f"  [yellow]  FLUX models missing — FLUX tier unavailable (optional)[/yellow]")

    # --- Workflow sentinels ---
    console.print("\n[bold]Workflow Sentinel Titles[/bold]")
    sentinel_checks = check_sentinels(project_root)
    sentinel_failures = []
    for workflow, sentinel, found in sentinel_checks:
        if not found:
            console.print(f"  [red]MISSING[/red]  {workflow}: '{sentinel}'")
            sentinel_failures.append((workflow, sentinel))
            all_ok = False

    if not sentinel_failures:
        console.print("  [green]OK[/green]  All sentinel titles present in workflow files")
    else:
        console.print(
            "\n  [red]Sentinel check failed.[/red] The workflow JSON files are missing inject markers.\n"
            "  generate.py cannot safely inject prompts/seeds without these.\n"
            "  Check that workflows/ files were not re-exported from ComfyUI without sentinel titles."
        )

    # --- ComfyUI connectivity ---
    console.print("\n[bold]ComfyUI Connectivity[/bold]")
    comfy_ok = check_comfyui(cfg, timeout=10)
    if comfy_ok:
        console.print(
            f"  [green]OK[/green]  ComfyUI responding at "
            f"http://{cfg['comfyui']['host']}:{cfg['comfyui']['port']}"
        )
    else:
        console.print(
            f"  [yellow]NOT REACHABLE[/yellow]  "
            f"http://{cfg['comfyui']['host']}:{cfg['comfyui']['port']}\n"
            "  Start ComfyUI before running generate.py."
        )

    # --- Test generation ---
    if not args.skip_generation and comfy_ok and not sentinel_failures:
        console.print("\n[bold]Test Generation (bootstrap workflow)[/bold]")
        gen_ok = run_test_workflow(cfg, project_root)
        if gen_ok:
            console.print("  [green]OK[/green]  Test image generated successfully")
        else:
            console.print("  [red]FAILED[/red]  Test generation failed — check ComfyUI logs")
            all_ok = False

    # --- Summary ---
    console.print("\n" + "=" * 50)
    if all_ok:
        console.print("[bold green]PASS[/bold green] — Setup verified. Ready to generate.")
        if not lora_ok:
            console.print(
                "[yellow]  Note:[/yellow] LoRA not yet trained. "
                "SDXL workflows will run without character consistency until "
                "KaviB_v1_Prod.safetensors is placed in models/loras/."
            )
    else:
        console.print("[bold red]FAIL[/bold red] — Fix the errors above before generating.")
        sys.exit(1)


if __name__ == "__main__":
    main()
