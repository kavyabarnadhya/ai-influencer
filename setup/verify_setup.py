import json
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from comfyui_api import ComfyUIClient

console = Console()

WORKFLOW_SENTINELS = {
    "bootstrap_seeds": ["_claude_inject_prompt", "_claude_inject_negative", "_claude_inject_seed", "_claude_inject_checkpoint", "_claude_inject_latent"],
    "bootstrap_ipadapter": ["_claude_inject_prompt", "_claude_inject_negative", "_claude_inject_seed", "_claude_inject_checkpoint", "_claude_inject_latent", "_claude_inject_ipadapter_image"],
    "t2i_sdxl_lora": ["_claude_inject_prompt", "_claude_inject_negative", "_claude_inject_seed", "_claude_inject_checkpoint", "_claude_inject_latent", "_claude_inject_lora"],
    "t2i_ipadapter": ["_claude_inject_prompt", "_claude_inject_negative", "_claude_inject_seed", "_claude_inject_checkpoint", "_claude_inject_latent", "_claude_inject_lora", "_claude_inject_ipadapter_image"],
    "flux_schnell": ["_claude_inject_prompt", "_claude_inject_seed", "_claude_inject_latent"],
    "flux_schnell_lora": ["_claude_inject_prompt", "_claude_inject_seed", "_claude_inject_latent", "_claude_inject_flux_lora"],
    "flux_img2img": ["_claude_inject_prompt", "_claude_inject_seed", "_claude_inject_init_image"],
    "flux_inpaint_person": ["_claude_inject_prompt", "_claude_inject_seed", "_claude_inject_bg_image", "_claude_inject_mask_image"],
    "faceswap_reactor": ["_claude_inject_source_image", "_claude_inject_target_image"],
}


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def get_workflow_titles(workflow_path: Path) -> set[str]:
    data = json.loads(workflow_path.read_text(encoding="utf-8"))
    return {node.get("_meta", {}).get("title", "") for node in data.values()}


def check_models(cfg: dict) -> list[tuple[str, bool, str]]:
    models_dir = Path(cfg["models"]["dir"])
    checks = [
        ("Checkpoint (Juggernaut XL)", models_dir / "checkpoints" / cfg["models"]["checkpoint"]),
        ("IP-Adapter Plus Face", models_dir / "ipadapter" / "sdxl_models" / cfg["models"]["ipadapter"]),
        ("FLUX GGUF Q4_K_S", models_dir / "unet" / cfg["models"]["flux_gguf_q4"]),
        ("FLUX GGUF Q3_K_S", models_dir / "unet" / cfg["models"]["flux_gguf_q3"]),
        ("FLUX VAE", models_dir / "vae" / cfg["models"]["flux_vae"]),
        ("CLIP-L", models_dir / "clip" / cfg["models"]["clip_l"]),
        ("T5-XXL fp8", models_dir / "clip" / cfg["models"]["t5xxl"]),
    ]
    # Check LoRA for each character
    for char_name, char_cfg in cfg.get("characters", {}).items():
        checks.append((
            f"LoRA ({char_name}: {char_cfg['lora']})",
            models_dir / "loras" / char_cfg["lora"],
        ))
        if char_cfg.get("flux_lora"):
            checks.append((
                f"FLUX LoRA ({char_name}: {char_cfg['flux_lora']})",
                models_dir / "loras" / char_cfg["flux_lora"],
            ))
    return [(name, path.exists(), str(path)) for name, path in checks]


def check_workflows(cfg: dict) -> list[tuple[str, bool, str]]:
    results = []
    wf_dir = ROOT / cfg["paths"]["workflows_dir"]
    for wf_name, sentinels in WORKFLOW_SENTINELS.items():
        path = wf_dir / f"{wf_name}.json"
        if not path.exists():
            results.append((wf_name, False, "file missing"))
            continue
        titles = get_workflow_titles(path)
        missing = [s for s in sentinels if s not in titles]
        if missing:
            results.append((wf_name, False, f"missing sentinels: {', '.join(missing)}"))
        else:
            results.append((wf_name, True, "all sentinels present"))
    return results


def check_character_files(cfg: dict) -> list[tuple[str, bool, str]]:
    results = []
    for char_name, char_cfg in cfg.get("characters", {}).items():
        base_prompt = ROOT / char_cfg["base_prompt_file"]
        results.append((f"{char_name}/base_prompt.txt", base_prompt.exists(), str(base_prompt.relative_to(ROOT))))
    return results


@click.command()
@click.option("--run-test", is_flag=True, help="Submit a minimal generation test to ComfyUI")
def main(run_test: bool):
    cfg = load_config()
    all_passed = True

    # Models
    model_table = Table(title="Model Files")
    model_table.add_column("Model", style="cyan")
    model_table.add_column("Status")
    model_table.add_column("Path", style="dim")
    for name, present, path in check_models(cfg):
        status = "[green]found[/green]" if present else "[yellow]missing (optional if not using this tier)[/yellow]"
        model_table.add_row(name, status, path)
    console.print(model_table)

    # Workflows
    wf_table = Table(title="Workflow Sentinels")
    wf_table.add_column("Workflow", style="cyan")
    wf_table.add_column("Status")
    wf_table.add_column("Detail", style="dim")
    for name, ok, detail in check_workflows(cfg):
        if not ok:
            all_passed = False
        status = "[green]ok[/green]" if ok else "[red]fail[/red]"
        wf_table.add_row(name, status, detail)
    console.print(wf_table)

    # ComfyUI connectivity
    comfy_cfg = cfg["comfyui"]
    client = ComfyUIClient(comfy_cfg["host"], comfy_cfg["port"])
    running = client.is_running()
    if not running:
        all_passed = False
    conn_status = "[green]reachable[/green]" if running else "[red]not running[/red]"
    console.print(f"\nComfyUI ({comfy_cfg['host']}:{comfy_cfg['port']}): {conn_status}")

    # Character files
    char_table = Table(title="Character Files")
    char_table.add_column("File", style="cyan")
    char_table.add_column("Status")
    char_table.add_column("Path", style="dim")
    for name, ok, path in check_character_files(cfg):
        if not ok:
            all_passed = False
        char_table.add_row(name, "[green]found[/green]" if ok else "[red]missing[/red]", path)
    console.print(char_table)

    if run_test and running:
        console.print("\n[cyan]Running minimal generation test...[/cyan]")
        from comfyui_api import inject_workflow_values, load_workflow
        import random
        wf = load_workflow(str(ROOT / cfg["paths"]["workflows_dir"] / "bootstrap_seeds.json"))
        # Use ananya base prompt for test
        ananya_cfg = cfg["characters"].get("ananya", {})
        test_prompt = "north indian woman, test image, photorealistic, editorial portrait"
        overrides = {
            "_claude_inject_prompt": {"inputs.text": test_prompt},
            "_claude_inject_negative": {"inputs.text": "worst quality"},
            "_claude_inject_seed": {"inputs.seed": random.randint(0, 9999), "inputs.steps": 4, "inputs.cfg": 1.0},
            "_claude_inject_latent": {"inputs.width": 512, "inputs.height": 512},
            "_claude_inject_checkpoint": {"inputs.ckpt_name": cfg["models"]["checkpoint"]},
        }
        try:
            patched = inject_workflow_values(wf, overrides)
            prompt_id = client.submit_workflow(patched)
            images = client.wait_for_completion(prompt_id, timeout=120)
            console.print(f"[green]Test generation produced {len(images)} image(s)[/green]")
        except Exception as e:
            console.print(f"[red]Test generation failed: {e}[/red]")
            all_passed = False

    if all_passed:
        console.print("\n[bold green]All checks passed. Pipeline ready.[/bold green]")
    else:
        console.print("\n[bold yellow]Some checks failed — review above before generating.[/bold yellow]")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
