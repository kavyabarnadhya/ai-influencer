import os
import random
import sys
from datetime import datetime
from pathlib import Path

import click
import yaml
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from comfyui_api import ComfyUIClient, ComfyUIError, inject_workflow_values, load_workflow

console = Console()
ROOT = Path(__file__).parent.parent


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_character(cfg: dict, character: str) -> dict:
    chars = cfg.get("characters", {})
    if character not in chars:
        available = list(chars.keys())
        console.print(f"[red]Unknown character '{character}'. Available: {available}[/red]")
        raise SystemExit(1)
    return chars[character]


JEWELRY_POOL = [
    "tiny understated gold stud earrings, barely visible thin gold chain necklace",
    "small matte gold hoop earrings, delicate thin gold pendant necklace",
    "tiny diamond stud earrings, subtle thin gold necklace",
    "small pearl stud earrings, no necklace",
    "small subtle gold drop earrings, delicate thin gold chain necklace",
    "tiny gold huggie earrings, barely visible thin gold necklace",
    "small silver stud earrings, no necklace",
    "no earrings, no necklace",
]

RING_BRACELET_POOL = [
    "thin subtle gold ring",
    "no rings, no bracelet",
    "no rings, no bracelet",
    "no rings, no bracelet",
]


def pick_jewelry() -> str:
    earring_necklace = random.choice(JEWELRY_POOL)
    ring_bracelet = random.choice(RING_BRACELET_POOL)
    return f"{earring_necklace}, {ring_bracelet}"


def build_prompt(base: str, user_prompt: str, trigger: str) -> str:
    jewelry = pick_jewelry()
    full = f"{base}, {jewelry}, {user_prompt}"
    if not full.startswith(trigger):
        full = f"{trigger}, {full}"
    return full


@click.command()
@click.option("--prompt", required=True, help="Scene description (no anatomy — LoRA handles that)")
@click.option("--count", default=1, show_default=True, help="Number of images to generate")
@click.option("--seed", default=None, type=int, help="Seed (random if omitted)")
@click.option("--workflow", default=None, help="Workflow name (overrides config default)")
@click.option("--rescue", is_flag=True, help="Low-VRAM mode: 768x1152, 24 steps, no FaceDetailer")
@click.option("--character", default="ananya", show_default=True, help="Character to generate [ananya|kavib]")
def main(prompt: str, count: int, seed: int | None, workflow: str | None, rescue: bool, character: str):
    cfg = load_config()
    char_cfg = load_character(cfg, character)
    comfy_cfg = cfg["comfyui"]
    gen_cfg = cfg["generation"]
    rescue_cfg = cfg["rescue_mode"]

    client = ComfyUIClient(comfy_cfg["host"], comfy_cfg["port"])
    if not client.is_running():
        console.print("[red]ComfyUI is not running. Start it first.[/red]")
        raise SystemExit(1)

    workflow_name = workflow or (rescue_cfg["workflow"] if rescue else gen_cfg["workflow"])
    workflow_path = ROOT / cfg["paths"]["workflows_dir"] / f"{workflow_name}.json"
    if not workflow_path.exists():
        console.print(f"[red]Workflow not found: {workflow_path}[/red]")
        raise SystemExit(1)

    base_prompt = (ROOT / char_cfg["base_prompt_file"]).read_text(encoding="utf-8").strip()
    full_prompt = build_prompt(base_prompt, prompt, char_cfg["trigger_word"])

    width = rescue_cfg["width"] if rescue else gen_cfg["width"]
    height = rescue_cfg["height"] if rescue else gen_cfg["height"]
    steps = rescue_cfg["steps"] if rescue else gen_cfg["steps"]

    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = ROOT / cfg["paths"]["output_dir"] / today / char_cfg["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    workflow_data = load_workflow(str(workflow_path))

    for i in range(count):
        img_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

        overrides = {
            "_claude_inject_prompt": {"inputs.text": full_prompt},
            "_claude_inject_negative": {"inputs.text": gen_cfg["negative_prompt"]},
            "_claude_inject_seed": {"inputs.seed": img_seed, "inputs.steps": steps, "inputs.cfg": gen_cfg["cfg"]},
            "_claude_inject_latent": {"inputs.width": width, "inputs.height": height},
            "_claude_inject_checkpoint": {"inputs.ckpt_name": cfg["models"]["checkpoint"]},
        }

        if workflow_name not in ("bootstrap_seeds", "flux_schnell"):
            overrides["_claude_inject_lora"] = {
                "inputs.lora_name": char_cfg["lora"],
                "inputs.strength_model": char_cfg["lora_strength"],
                "inputs.strength_clip": char_cfg["lora_strength"],
            }

        patched = inject_workflow_values(workflow_data, overrides)

        console.print(f"[cyan]Generating image {i + 1}/{count} (seed {img_seed})...[/cyan]")
        try:
            prompt_id = client.submit_workflow(patched)
            images = client.wait_for_completion(prompt_id, timeout=comfy_cfg["timeout"])
        except ComfyUIError as e:
            console.print(f"[red]Generation failed: {e}[/red]")
            raise SystemExit(1)

        for img_meta in images:
            img_bytes = client.download_image(
                img_meta["filename"], img_meta.get("subfolder", ""), img_meta.get("type", "output")
            )
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"{character}_{today}_{timestamp}_{img_seed}.png"
            dest = out_dir / filename
            dest.write_bytes(img_bytes)
            console.print(f"[green]Saved:[/green] {dest}")


if __name__ == "__main__":
    main()
