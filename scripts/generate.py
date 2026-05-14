import os
import random
import sys
from datetime import datetime
from pathlib import Path

import click
import yaml
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from comfyui_api import ComfyUIClient, ComfyUIError, inject_workflow_values, load_workflow, find_comfyui_port
from presets import load_preset, render_prompt, merge_singleshot

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
    return any(tok in lowered for tok in ("empty scene", "no people", "no humans", "no person", "ambient shot", "background only"))


@click.command()
@click.option("--prompt", default=None, help="Scene description (no anatomy — LoRA handles that)")
@click.option("--preset", default=None, help="Named preset from character/<char>/presets.yaml")
@click.option("--outfit", default=None, help="Outfit description — fills {outfit} in preset template (CLI wins over preset default)")
@click.option("--hair", default=None, help="Hair description — fills {hair} in preset template (CLI wins over preset default)")
@click.option("--scene", default=None, help="Scene description — fills {scene} in preset template (CLI wins over preset default)")
@click.option("--image", help="Optional reference image path for IP-Adapter (e.g. a FLUX background)")
@click.option("--count", default=1, show_default=True, help="Number of images to generate")
@click.option("--seed", default=None, type=int, help="Seed (random if omitted)")
@click.option("--workflow", default=None, help="Workflow name (overrides config default)")
@click.option("--rescue", is_flag=True, help="Low-VRAM mode: 768x1152, 24 steps, no FaceDetailer")
@click.option("--reel-anchor", is_flag=True, help="Generate a vertical still for image-to-video reels")
@click.option("--character", default="ananya", show_default=True, help="Character to generate [ananya|kavib]")
@click.option("--upscale", is_flag=True, help="4x upscale after detailing using 4x-UltraSharp")
@click.option("--pose", default=None, help="Path to pose reference image for ControlNet OpenPose")
@click.option("--steps", default=None, type=int, help="Override step count (e.g. 15 for draft, 30 for final)")
def main(prompt: str | None, preset: str | None, outfit: str | None, hair: str | None, scene: str | None, image: str | None, count: int, seed: int | None, workflow: str | None, rescue: bool, reel_anchor: bool, character: str, upscale: bool, pose: str | None, steps: int | None):
    if not prompt and not preset:
        console.print("[red]Provide either --prompt or --preset[/red]")
        raise SystemExit(1)

    cfg = load_config()
    char_cfg = load_character(cfg, character)
    comfy_cfg = cfg["comfyui"]

    # Resolve preset — CLI flags win over preset values
    resolved = None
    if preset:
        try:
            preset_data = load_preset(character, preset)
        except (FileNotFoundError, KeyError) as e:
            console.print(f"[red]{e}[/red]")
            raise SystemExit(1)
        if preset_data.get("kind") == "carousel":
            console.print(f"[red]'{preset}' is a carousel preset — use generate_carousel.py --preset {preset}[/red]")
            raise SystemExit(1)
        resolved = merge_singleshot(preset_data, dict(
            workflow=workflow, outfit=outfit, hair=hair, scene=scene,
            pose=pose, image=image, steps=steps, seed=seed,
        ))
        workflow = resolved["workflow"] or workflow
        pose     = resolved["pose"]     or pose
        image    = resolved["face_ref"] or image
        if resolved["steps"] is not None:
            steps = resolved["steps"]
        if resolved["seed"] is not None:
            seed = resolved["seed"]
        prompt = render_prompt(
            preset_data["prompt_template"],
            outfit=resolved["outfit"],
            hair=resolved["hair"],
            scene=resolved["scene"],
        )
    gen_cfg = cfg["generation"]
    rescue_cfg = cfg["rescue_mode"]

    host = comfy_cfg["host"]
    port = find_comfyui_port(host, [comfy_cfg["port"], 8000, 8188, 8002])
    if port is None:
        console.print("[red]ComfyUI is not running. Start it first.[/red]")
        raise SystemExit(1)
    if port != comfy_cfg["port"]:
        console.print(f"[yellow]ComfyUI found on port {port} (config says {comfy_cfg['port']})[/yellow]")
    client = ComfyUIClient(host, port)

    # If an image is provided but no workflow is specified, default to t2i_img2img
    if pose and upscale and not workflow:
        workflow_name = "t2i_sdxl_controlnet_upscale"
    elif pose and not workflow:
        workflow_name = "t2i_sdxl_controlnet"
    elif upscale and not workflow:
        workflow_name = "t2i_sdxl_upscale"
    elif image and not workflow:
        workflow_name = "t2i_img2img"
    else:
        workflow_name = workflow or (rescue_cfg["workflow"] if rescue else gen_cfg["workflow"])

    workflow_path = ROOT / cfg["paths"]["workflows_dir"] / f"{workflow_name}.json"
    if not workflow_path.exists():
        console.print(f"[red]Workflow not found: {workflow_path}[/red]")
        raise SystemExit(1)

    is_flux = is_flux_workflow(workflow_name)
    background_only = is_background_prompt(prompt)
    base_prompt = (ROOT / char_cfg["base_prompt_file"]).read_text(encoding="utf-8").strip()
    full_prompt = prompt if background_only else build_prompt(
        base_prompt,
        prompt,
        char_cfg["trigger_word"],
        include_base=not is_flux,
    )
    if is_flux and not background_only:
        full_prompt += ", fair complexion, soft front lighting, no text, no watermark"

    reels_cfg = cfg.get("reels", {})
    if reel_anchor:
        width = reels_cfg.get("anchor_width", gen_cfg["width"])
        height = reels_cfg.get("anchor_height", gen_cfg["height"])
    else:
        width = rescue_cfg["width"] if rescue else gen_cfg["width"]
        height = rescue_cfg["height"] if rescue else gen_cfg["height"]
    steps = steps if steps is not None else (rescue_cfg["steps"] if rescue else gen_cfg["steps"])

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

    uploaded_pose = None
    if pose:
        pose_path = Path(pose)
        if not pose_path.exists():
            console.print(f"[red]Pose reference image not found: {pose}[/red]")
            raise SystemExit(1)
        console.print(f"[dim]Uploading pose reference: {pose_path.name}...[/dim]")
        uploaded_pose = client.upload_image(str(pose_path))

    # Pre-patch constant workflow values once outside the loop
    effective_steps = 4 if is_flux else steps
    current_cfg = 1.0 if is_flux else gen_cfg["cfg"]

    base_overrides = {
        "_claude_inject_prompt": {"inputs.text": full_prompt},
        "_claude_inject_negative": {"inputs.text": gen_cfg["negative_prompt"]},
        "_claude_inject_seed": {"inputs.steps": effective_steps, "inputs.cfg": current_cfg},
        "_claude_inject_latent": {"inputs.width": width, "inputs.height": height},
        "_claude_inject_checkpoint": {"inputs.ckpt_name": cfg["models"]["checkpoint"]},
    }

    if uploaded_filename:
        base_overrides["_claude_inject_ipadapter_image"] = {"inputs.image": uploaded_filename}

    if uploaded_pose:
        base_overrides["_claude_inject_controlnet_image"] = {"inputs.image": uploaded_pose}
        base_overrides["_claude_inject_controlnet"] = {"inputs.control_net_name": cfg["models"]["controlnet_openpose"]}

    if is_flux and not background_only:
        flux_lora = char_cfg.get("flux_lora")
        if flux_lora:
            flux_strength = char_cfg.get("flux_lora_strength", char_cfg.get("lora_strength", 0.85))
            base_overrides["_claude_inject_flux_lora"] = {
                "inputs.lora_name": flux_lora,
                "inputs.strength_model": flux_strength,
                "inputs.strength_clip": flux_strength,
            }

    if workflow_name not in ("bootstrap_seeds", "flux_schnell") and not background_only:
        lora_strength = (resolved or {}).get("lora_strength") or char_cfg["lora_strength"]
        base_overrides["_claude_inject_lora"] = {
            "inputs.lora_name": char_cfg["lora"],
            "inputs.strength_model": lora_strength,
            "inputs.strength_clip": lora_strength,
        }

    # Apply preset overrides for IPAdapter and ControlNet (silently skipped if node absent)
    if resolved:
        if resolved.get("ipadapter_strength") is not None:
            base_overrides.setdefault("_claude_inject_ipadapter_strength", {})["inputs.weight"] = resolved["ipadapter_strength"]
        if resolved.get("ipadapter_start_at") is not None:
            base_overrides.setdefault("_claude_inject_ipadapter_strength", {})["inputs.start_at"] = resolved["ipadapter_start_at"]
        if resolved.get("ipadapter_end_at") is not None:
            base_overrides.setdefault("_claude_inject_ipadapter_strength", {})["inputs.end_at"] = resolved["ipadapter_end_at"]
        if resolved.get("denoise") is not None:
            base_overrides.setdefault("_claude_inject_seed", {})["inputs.denoise"] = resolved["denoise"]
        if resolved.get("controlnet_strength") is not None:
            base_overrides.setdefault("ControlNet Apply", {})["inputs.strength"] = resolved["controlnet_strength"]

    if upscale:
        base_overrides["_claude_inject_upscaler"] = {
            "inputs.model_name": cfg["models"]["upscaler"],
        }

    workflow_data = inject_workflow_values(workflow_data, base_overrides)

    pending_prompts: list[tuple[str, int]] = []

    for i in range(count):
        img_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

        overrides = {
            "_claude_inject_seed": {"inputs.seed": img_seed},
        }
        # Performance Optimization: Skip cache propagation on the final injection
        # to avoid an extra dictionary copy in client.submit_workflow().
        patched = inject_workflow_values(workflow_data, overrides, propagate_cache=False)

        if is_flux and not background_only and i == 0:
            _flux_prompt_hints(full_prompt)
        console.print(f"[cyan]Submitting image {i + 1}/{count} (seed {img_seed})...[/cyan]")
        try:
            prompt_id = client.submit_workflow(patched)
            pending_prompts.append((prompt_id, img_seed))
        except ComfyUIError as e:
            console.print(f"[red]Submission failed: {e}[/red]")
            raise SystemExit(1)

    # Wait for and download results
    for i, (prompt_id, img_seed) in enumerate(pending_prompts):
        console.print(f"[cyan]Waiting for image {i + 1}/{count} (seed {img_seed})...[/cyan]")
        try:
            images = client.wait_for_completion(prompt_id, timeout=comfy_cfg["timeout"])
        except ComfyUIError as e:
            console.print(f"[red]Generation failed for seed {img_seed}: {e}[/red]")
            continue

        for img_meta in images:
            img_bytes = client.download_image(
                img_meta["filename"], img_meta.get("subfolder", ""), img_meta.get("type", "output")
            )
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"{character}_{today}_{timestamp}_{img_seed}.png"
            dest = out_dir / filename
            dest.write_bytes(img_bytes)
            print(f"Saved: {dest}")
            client.delete_output_image(
                img_meta["filename"], img_meta.get("subfolder", ""), comfy_cfg.get("output_dir")
            )

    # Log generation metadata
    import json
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "preset": preset,
        "character": character,
        "workflow": workflow_name,
        "prompt": full_prompt,
        "outfit": (resolved or {}).get("outfit"),
        "hair": (resolved or {}).get("hair"),
        "scene": (resolved or {}).get("scene"),
        "lora_strength": (resolved or {}).get("lora_strength") or char_cfg["lora_strength"],
        "ipadapter_strength": (resolved or {}).get("ipadapter_strength"),
        "ipadapter_start_at": (resolved or {}).get("ipadapter_start_at"),
        "denoise": (resolved or {}).get("denoise"),
        "controlnet_strength": (resolved or {}).get("controlnet_strength"),
        "pose": pose,
        "face_ref": image,
        "steps": effective_steps,
        "cfg": current_cfg,
        "width": width,
        "height": height,
        "seeds": [s for _, s in pending_prompts],
    }
    log_path = ROOT / "logs" / "prompt_history.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


if __name__ == "__main__":
    main()
