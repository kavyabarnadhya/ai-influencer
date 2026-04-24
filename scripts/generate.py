#!/usr/bin/env python3
"""Main image generation CLI.

Usage:
    python scripts/generate.py --prompt "sitting at a sunlit cafe with matcha latte"
    python scripts/generate.py --prompt "..." --workflow t2i_sdxl_lora --count 4 --seed 42
    python scripts/generate.py --prompt "..." --rescue   # low-VRAM safe preset
"""

import copy
import datetime
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

try:
    import click
    import yaml
    from rich.console import Console
    from comfyui_api import ComfyUIClient, ComfyUIError
except ImportError:
    print("Missing dependencies. Run: pip install click PyYAML rich")
    sys.exit(1)

console = Console()


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_workflow(workflow_name: str) -> dict:
    p = PROJECT_ROOT / "workflows" / f"{workflow_name}.json"
    if not p.exists():
        raise FileNotFoundError(f"Workflow not found: {p}")
    return json.loads(p.read_text())


def inject_into_workflow(workflow: dict, injections: dict) -> dict:
    """Walk workflow nodes and inject values via stable sentinel titles.

    Each node carries a _meta.title sentinel. The function below maps
    each sentinel to one or more (field, injection_key) pairs so a single
    KSampler node can receive seed + steps + cfg in one pass.

    Never uses raw node IDs — those change on every ComfyUI export.
    """
    # Each entry: sentinel_title -> list of (input_field, injection_key)
    SENTINEL_MAP: dict[str, list[tuple[str, str]]] = {
        "_claude_inject_prompt":      [("text",           "_claude_inject_prompt")],
        "_claude_inject_negative":    [("text",           "_claude_inject_negative")],
        "_claude_inject_checkpoint":  [("ckpt_name",      "_claude_inject_checkpoint")],
        "_claude_inject_lora":        [("lora_name",      "_claude_inject_lora"),
                                       ("strength_model", "_claude_inject_lora_strength"),
                                       ("strength_clip",  "_claude_inject_lora_strength")],
        "_claude_inject_latent":      [("width",          "_claude_inject_width"),
                                       ("height",         "_claude_inject_height")],
        # KSampler/KSamplerAdvanced: one sentinel injects seed + steps + cfg.
        # Both "seed" (KSampler) and "noise_seed" (KSamplerAdvanced) are set;
        # each node type only uses the field relevant to its schema.
        "_claude_inject_seed":        [("seed",           "_claude_inject_seed"),
                                       ("noise_seed",     "_claude_inject_seed"),
                                       ("steps",          "_claude_inject_steps"),
                                       ("cfg",            "_claude_inject_cfg")],
    }

    wf = copy.deepcopy(workflow)
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        title = node.get("_meta", {}).get("title", "")
        if title not in SENTINEL_MAP:
            continue
        for field, key in SENTINEL_MAP[title]:
            if key in injections:
                node.setdefault("inputs", {})[field] = injections[key]
    return wf


def build_injections(cfg: dict, prompt: str, seed: int, steps: int, cfg_scale: float,
                     width: int, height: int, adetailer: bool) -> dict:
    lora_trigger = cfg["character"]["lora_trigger"]
    full_prompt = f"{lora_trigger}, {prompt}"

    injections = {
        "_claude_inject_prompt":       full_prompt,
        "_claude_inject_negative":     (
            "blurry, extra fingers, malformed hands, deformed eyes, bad anatomy, "
            "duplicate person, watermark, text, low quality"
        ),
        "_claude_inject_seed":         seed,
        "_claude_inject_steps":        steps,
        "_claude_inject_cfg":          cfg_scale,
        "_claude_inject_width":        width,
        "_claude_inject_height":       height,
        "_claude_inject_checkpoint":   cfg["models"]["sdxl_checkpoint"],
        "_claude_inject_lora":         cfg["models"]["lora"],
        "_claude_inject_lora_strength": cfg["models"]["lora_strength"],
    }
    return injections


def write_sidecar(image_path: Path, meta: dict) -> None:
    sidecar = image_path.with_suffix(".json")
    with open(sidecar, "w") as f:
        json.dump(meta, f, indent=2)


@click.command()
@click.option("--prompt", required=True, help="Scene description (base_prompt auto-prepended)")
@click.option("--workflow", default="t2i_sdxl_lora", show_default=True,
              help="Workflow name (without .json)")
@click.option("--count", default=1, show_default=True, help="Number of images to generate")
@click.option("--seed", default=-1, show_default=True, help="Seed (-1 for random)")
@click.option("--steps", default=None, type=int, help="Sampling steps (overrides config)")
@click.option("--cfg", "cfg_scale", default=None, type=float, help="CFG scale (overrides config)")
@click.option("--width", default=None, type=int, help="Image width (overrides config)")
@click.option("--height", default=None, type=int, help="Image height (overrides config)")
@click.option("--output-dir", default=None, help="Output directory (overrides config)")
@click.option("--no-adetailer", is_flag=True, help="Disable ADetailer/FaceDetailer pass")
@click.option("--rescue", is_flag=True,
              help="Low-VRAM safe preset: 768x1152, 24 steps, count=1, ADetailer off")
@click.option("--config", "config_path", default="config.yaml", show_default=True)
def main(prompt, workflow, count, seed, steps, cfg_scale, width, height,
         output_dir, no_adetailer, rescue, config_path):
    cfg = load_config(PROJECT_ROOT / config_path)
    gen = cfg["generation"]

    if rescue:
        console.print("[yellow]--rescue mode:[/yellow] 768×1152, 24 steps, count=1, ADetailer off")
        width = 768
        height = 1152
        steps = 24
        count = 1
        no_adetailer = True

    steps = steps or gen["default_steps"]
    cfg_scale = cfg_scale or gen["default_cfg"]
    width = width or gen["default_width"]
    height = height or gen["default_height"]
    out_base = Path(output_dir or gen["output_dir"])
    date_str = datetime.date.today().isoformat()
    out_dir = out_base / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    # Suffix workflow name with _nofacedetail if adetailer disabled and workflow supports it
    effective_workflow = workflow
    if no_adetailer and "lora" in workflow:
        effective_workflow = workflow  # still load same JSON; FaceDetailer node is bypassed via sentinel

    try:
        wf_template = load_workflow(effective_workflow)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    client = ComfyUIClient(host=cfg["comfyui"]["host"], port=cfg["comfyui"]["port"])
    console.print(f"Waiting for ComfyUI at {cfg['comfyui']['host']}:{cfg['comfyui']['port']}...")
    try:
        client.wait_for_ready(timeout=cfg["comfyui"]["startup_timeout"])
    except ComfyUIError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    console.print(
        f"\n[bold]Generating {count} image(s)[/bold]\n"
        f"  Workflow : {effective_workflow}\n"
        f"  Prompt   : {cfg['character']['lora_trigger']}, {prompt}\n"
        f"  Size     : {width}×{height}  Steps: {steps}  CFG: {cfg_scale}\n"
        f"  Output   : {out_dir}\n"
    )

    for i in range(count):
        actual_seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
        injections = build_injections(cfg, prompt, actual_seed, steps, cfg_scale, width, height, not no_adetailer)
        wf = inject_into_workflow(wf_template, injections)

        console.print(f"  [{i+1}/{count}] seed={actual_seed} ...", end=" ")
        try:
            images = client.submit_and_wait(wf, str(out_dir))
            for img_path in images:
                meta = {
                    "prompt": prompt,
                    "full_prompt": injections["_claude_inject_prompt"],
                    "negative_prompt": injections["_claude_inject_negative"],
                    "workflow": effective_workflow,
                    "seed": actual_seed,
                    "steps": steps,
                    "cfg": cfg_scale,
                    "width": width,
                    "height": height,
                    "adetailer": not no_adetailer,
                    "lora": cfg["models"]["lora"],
                    "lora_strength": cfg["models"]["lora_strength"],
                    "date": date_str,
                }
                write_sidecar(img_path, meta)
            console.print(f"[green]OK[/green] → {', '.join(p.name for p in images)}")
        except ComfyUIError as e:
            console.print(f"[red]FAILED[/red]: {e}")

    console.print(f"\n[green]Done.[/green] Images saved to {out_dir}")


if __name__ == "__main__":
    main()
