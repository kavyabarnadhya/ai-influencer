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


def build_prompt(base: str, user_prompt: str, trigger: str, include_base: bool = True) -> str:
    jewelry = pick_jewelry()
    if not include_base:
        return f"{trigger}, {user_prompt}, {jewelry}"

    # Move user_prompt to the front (after trigger) to give it maximum weight
    # and strip the trigger from the base if it exists to avoid double-triggering.
    clean_base = base.replace(trigger, "").strip().strip(",")
    full = f"{trigger}, {user_prompt}, {jewelry}, {clean_base}"
    return full


def is_flux_workflow(workflow_name: str) -> bool:
    return workflow_name.startswith("flux")


def _flux_prompt_hints(prompt: str) -> None:
    hints = []
    p = prompt.lower()
    if "sharp detailed face" not in p:
        hints.append("'sharp detailed face' — improves identity on all shots")
    if any(w in p for w in ("full-body", "full body")) and "35mm" not in p:
        hints.append("'35mm portrait lens, face in focus' — prevents identity drift on full-body shots")
    if any(w in p for w in ("premium", "evening", "cocktail", "gown", "vanity")) and "editorial" not in p:
        hints.append("'editorial fashion photography' — elevates premium scenes")
    if hints:
        console.print("[yellow]Prompt tips:[/yellow]")
        for h in hints:
            console.print(f"[yellow]  + {h}[/yellow]")


def is_background_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    return "empty scene" in lowered or "no people" in lowered or "no humans" in lowered


@click.command()
@click.option("--prompt", required=True, help="Scene description (no anatomy — LoRA handles that)")
@click.option("--image", help="Optional reference image path for IP-Adapter (e.g. a FLUX background)")
@click.option("--count", default=1, show_default=True, help="Number of images to generate")
@click.option("--seed", default=None, type=int, help="Seed (random if omitted)")
@click.option("--workflow", default=None, help="Workflow name (overrides config default)")
@click.option("--rescue", is_flag=True, help="Low-VRAM mode: 768x1152, 24 steps, no FaceDetailer")
@click.option("--reel-anchor", is_flag=True, help="Generate a vertical still for image-to-video reels")
@click.option("--character", default="ananya", show_default=True, help="Character to generate [ananya|kavib]")
def main(prompt: str, image: str | None, count: int, seed: int | None, workflow: str | None, rescue: bool, reel_anchor: bool, character: str):
    cfg = load_config()
    char_cfg = load_character(cfg, character)
    comfy_cfg = cfg["comfyui"]
    gen_cfg = cfg["generation"]
    rescue_cfg = cfg["rescue_mode"]

    client = ComfyUIClient(comfy_cfg["host"], comfy_cfg["port"])
    if not client.is_running():
        console.print("[red]ComfyUI is not running. Start it first.[/red]")
        raise SystemExit(1)

    # If an image is provided but no workflow is specified, default to t2i_img2img
    if image and not workflow:
        workflow_name = "t2i_img2img"
    else:
        workflow_name = workflow or (rescue_cfg["workflow"] if rescue else gen_cfg["workflow"])

    workflow_path = ROOT / cfg["paths"]["workflows_dir"] / f"{workflow_name}.json"
    if not workflow_path.exists():
        console.print(f"[red]Workflow not found: {workflow_path}[/red]")
        raise SystemExit(1)

    is_flux = is_flux_workflow(workflow_name)
    background_only = is_flux and is_background_prompt(prompt)
    base_prompt = (ROOT / char_cfg["base_prompt_file"]).read_text(encoding="utf-8").strip()
    full_prompt = prompt if background_only else build_prompt(
        base_prompt,
        prompt,
        char_cfg["trigger_word"],
        include_base=not is_flux,
    )
    if is_flux:
        full_prompt += ", fair complexion, soft front lighting, no text, no watermark"

    reels_cfg = cfg.get("reels", {})
    if reel_anchor:
        width = reels_cfg.get("anchor_width", gen_cfg["width"])
        height = reels_cfg.get("anchor_height", gen_cfg["height"])
    else:
        width = rescue_cfg["width"] if rescue else gen_cfg["width"]
        height = rescue_cfg["height"] if rescue else gen_cfg["height"]
    steps = rescue_cfg["steps"] if rescue else gen_cfg["steps"]

    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = ROOT / cfg["paths"]["output_dir"] / today / char_cfg["output_subdir"]
    if reel_anchor:
        out_dir = out_dir / reels_cfg.get("anchor_subdir", "reels/anchors")
    out_dir.mkdir(parents=True, exist_ok=True)

    workflow_data = load_workflow(str(workflow_path))

    # Handle image upload if provided
    uploaded_filename = None
    if image:
        img_path = Path(image)
        if not img_path.exists():
            console.print(f"[red]Reference image not found: {image}[/red]")
            raise SystemExit(1)
        console.print(f"[dim]Uploading reference image: {img_path.name}...[/dim]")
        uploaded_filename = client.upload_image(str(img_path))

    for i in range(count):
        img_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

        current_steps = 4 if is_flux else steps
        current_cfg = 1.0 if is_flux else gen_cfg["cfg"]

        overrides = {
            "_claude_inject_prompt": {"inputs.text": full_prompt},
            "_claude_inject_negative": {"inputs.text": gen_cfg["negative_prompt"]},
            "_claude_inject_seed": {"inputs.seed": img_seed, "inputs.steps": current_steps, "inputs.cfg": current_cfg},
            "_claude_inject_latent": {"inputs.width": width, "inputs.height": height},
            "_claude_inject_checkpoint": {"inputs.ckpt_name": cfg["models"]["checkpoint"]},
        }

        if uploaded_filename:
            overrides["_claude_inject_ipadapter_image"] = {"inputs.image": uploaded_filename}

        if is_flux and not background_only:
            flux_lora = char_cfg.get("flux_lora")
            if flux_lora:
                flux_strength = char_cfg.get("flux_lora_strength", char_cfg.get("lora_strength", 0.85))
                overrides["_claude_inject_flux_lora"] = {
                    "inputs.lora_name": flux_lora,
                    "inputs.strength_model": flux_strength,
                    "inputs.strength_clip": flux_strength,
                }

        if workflow_name not in ("bootstrap_seeds", "flux_schnell"):
            overrides["_claude_inject_lora"] = {
                "inputs.lora_name": char_cfg["lora"],
                "inputs.strength_model": char_cfg["lora_strength"],
                "inputs.strength_clip": char_cfg["lora_strength"],
            }

        patched = inject_workflow_values(workflow_data, overrides)

        if is_flux and i == 0:
            _flux_prompt_hints(prompt)
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
