import random
import sys
from datetime import datetime
from pathlib import Path

import click
import yaml
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from comfyui_api import ComfyUIClient, ComfyUIError, inject_workflow_values, load_workflow, find_comfyui_port

console = Console()
ROOT = Path(__file__).parent.parent

SLIDE_ROLES = {
    "wide": "full body shot, wide angle, sharp background, environmental storytelling, soft smile, facing camera, weight on one leg",
    "medium": "medium shot, three quarter view, subtle smile, body turned slightly, looking over shoulder",
    "close": "close up portrait, soft smile, sharp focus on face, direct eye contact, chin slightly down",
    "hands": "close up of hands, fabric texture detail, jewelry detail, no face visible, sharp focus",
    "ambient": "no person, background only, ambient shot, interior detail, sharp focus, no blur",
}

DEFAULT_SLIDE_SEQUENCE = ["wide", "medium", "close"]

POSE_FILENAMES = {
    "wide": "wide.png",
    "medium": "medium.png",
    "close": "close.png",
}
FOUR_SLIDE_SEQUENCE = ["wide", "medium", "close", "ambient"]

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


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_character(cfg: dict, character: str) -> dict:
    chars = cfg.get("characters", {})
    if character not in chars:
        console.print(f"[red]Unknown character '{character}'. Available: {list(chars.keys())}[/red]")
        raise SystemExit(1)
    return chars[character]


def pick_jewelry() -> str:
    return f"{random.choice(JEWELRY_POOL)}, {random.choice(RING_BRACELET_POOL)}"


def is_detail_slide(role: str) -> bool:
    return role in ("hands", "ambient")


# Higher denoise = more pose freedom but more outfit drift
# medium: modest change from anchor; close: face-forward requires more departure
IMG2IMG_DENOISE = {
    "medium": 0.68,
    "close": 0.75,
    "hands": 0.75,
    "wide": 0.55,
}


def build_slide_prompt(scene: str, role: str, char_cfg: dict, base_prompt: str, carousel_cfg: dict, outfit: str = "", hair: str = "", jewelry: str = "") -> str:
    angle_tokens = SLIDE_ROLES[role]
    trigger = char_cfg["trigger_word"]
    body_tokens = carousel_cfg["body_tokens"]
    sharp_tokens = carousel_cfg["sharp_background_positive"]

    if is_detail_slide(role):
        return f"{scene}, {angle_tokens}"

    clean_base = base_prompt.replace(trigger, "").strip().strip(",")
    parts = [trigger, scene]
    if outfit:
        parts.append(outfit)
    if hair:
        parts.append(hair)
    parts += [angle_tokens, body_tokens, sharp_tokens, "symmetrical eyes, natural eyes, realistic eyes"]
    if jewelry:
        parts.append(jewelry)
    parts.append(clean_base)
    return ", ".join(p for p in parts if p)


def build_negative(base_negative: str, carousel_cfg: dict) -> str:
    return f"{base_negative}, {carousel_cfg['low_bokeh_negative']}"


def get_slide_sequence(slide_count: int) -> list[str]:
    if slide_count == 1:
        return ["medium"]
    if slide_count == 2:
        return ["wide", "medium"]
    if slide_count == 3:
        return DEFAULT_SLIDE_SEQUENCE
    if slide_count == 4:
        return FOUR_SLIDE_SEQUENCE
    # 5+: repeat medium/close before ambient
    seq = ["wide"] + ["medium", "close"] * ((slide_count - 2) // 2)
    seq = seq[:slide_count - 1] + ["ambient"]
    return seq[:slide_count]


@click.command()
@click.option("--scene", required=True, help="Location + lighting only (e.g. 'rooftop, golden hour, Mumbai skyline')")
@click.option("--outfit", required=True, help="Clothing locked across all model slides (e.g. 'burgundy silk saree')")
@click.option("--hair", default="dark hair neatly styled", show_default=True, help="Hair descriptor locked across all model slides")
@click.option("--slides", default=3, show_default=True, help="Number of slides (3-4 recommended)")
@click.option("--name", required=True, help="Carousel name — used for output subfolder (snake_case)")
@click.option("--character", default="ananya", show_default=True, help="Character [ananya|kavib]")
@click.option("--seed", default=None, type=int, help="Base seed (each slide gets seed+slide_index for variety)")
@click.option("--steps", default=None, type=int, help="Override step count")
@click.option("--face-ref", default=None, help="Path to face reference image for IP-Adapter (improves face consistency)")
@click.option("--poses-dir", default=None, help="Directory with pose images named wide.png, medium.png, close.png for ControlNet pose guidance")
def main(scene: str, outfit: str, hair: str, slides: int, name: str, character: str, seed: int | None, steps: int | None, face_ref: str | None, poses_dir: str | None):
    cfg = load_config()
    char_cfg = load_character(cfg, character)
    gen_cfg = cfg["generation"]
    carousel_cfg = cfg["carousel"]
    comfy_cfg = cfg["comfyui"]

    host = comfy_cfg["host"]
    port = find_comfyui_port(host, [comfy_cfg["port"], 8000, 8188, 8002])
    if port is None:
        console.print("[red]ComfyUI is not running. Start it first.[/red]")
        raise SystemExit(1)
    if port != comfy_cfg["port"]:
        console.print(f"[yellow]ComfyUI found on port {port}[/yellow]")
    client = ComfyUIClient(host, port)

    if poses_dir and face_ref:
        anchor_workflow_name = "t2i_sdxl_lora_ipadapter_controlnet"
        img2img_workflow_name = "t2i_sdxl_lora_img2img_ipadapter_controlnet"
    elif face_ref:
        anchor_workflow_name = "t2i_sdxl_lora_ipadapter"
        img2img_workflow_name = "t2i_sdxl_lora_img2img_ipadapter"
    else:
        anchor_workflow_name = gen_cfg["workflow"]
        img2img_workflow_name = "t2i_img2img"
    for wn in (anchor_workflow_name, img2img_workflow_name):
        wp = ROOT / cfg["paths"]["workflows_dir"] / f"{wn}.json"
        if not wp.exists():
            console.print(f"[red]Workflow not found: {wp}[/red]")
            raise SystemExit(1)

    uploaded_face = None
    if face_ref:
        face_path = Path(face_ref)
        if not face_path.exists():
            console.print(f"[red]Face reference image not found: {face_ref}[/red]")
            raise SystemExit(1)
        console.print(f"[dim]Uploading face reference: {face_path.name}...[/dim]")
        uploaded_face = client.upload_image(str(face_path))

    base_prompt = (ROOT / char_cfg["base_prompt_file"]).read_text(encoding="utf-8").strip()
    negative = build_negative(gen_cfg["negative_prompt"], carousel_cfg)
    effective_steps = steps if steps is not None else gen_cfg["steps"]
    base_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = ROOT / cfg["paths"]["output_dir"] / today / char_cfg["output_subdir"] / f"carousel_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    slide_sequence = get_slide_sequence(slides)
    locked_jewelry = pick_jewelry()
    console.print(f"[bold green]Carousel: {name}[/bold green] — {len(slide_sequence)} slides: {slide_sequence}")
    console.print(f"[dim]Scene: {scene}[/dim]")
    console.print(f"[dim]Jewelry: {locked_jewelry}[/dim]")

    info_lines = [f"# Carousel: {name}", f"Scene: {scene}", f"Generated: {today}", f"Slides: {len(slide_sequence)}", ""]

    anchor_image_path: Path | None = None
    uploaded_anchor: str | None = None

    for i, role in enumerate(slide_sequence):
        slide_num = i + 1
        img_seed = base_seed + i
        slide_prompt = build_slide_prompt(scene, role, char_cfg, base_prompt, carousel_cfg, outfit=outfit, hair=hair, jewelry=locked_jewelry)

        # img2img anchor disabled when poses_dir set — ControlNet+prompt+IPAdapter handle consistency instead
        use_img2img = (i > 0) and not is_detail_slide(role) and anchor_image_path is not None and not poses_dir
        current_workflow_name = img2img_workflow_name if use_img2img else anchor_workflow_name
        workflow_data = load_workflow(str(ROOT / cfg["paths"]["workflows_dir"] / f"{current_workflow_name}.json"))

        overrides = {
            "_claude_inject_prompt": {"inputs.text": slide_prompt},
            "_claude_inject_negative": {"inputs.text": negative},
            "_claude_inject_seed": {"inputs.seed": img_seed, "inputs.steps": effective_steps, "inputs.cfg": gen_cfg["cfg"]},
            "_claude_inject_checkpoint": {"inputs.ckpt_name": cfg["models"]["checkpoint"]},
            "_claude_inject_lora": {
                "inputs.lora_name": char_cfg["lora"],
                "inputs.strength_model": char_cfg["lora_strength"],
                "inputs.strength_clip": char_cfg["lora_strength"],
            },
        }

        if not use_img2img:
            overrides["_claude_inject_latent"] = {"inputs.width": gen_cfg["width"], "inputs.height": gen_cfg["height"]}

        if use_img2img and uploaded_anchor:
            overrides["_claude_inject_init_image"] = {"inputs.image": uploaded_anchor}
            overrides["_claude_inject_seed"]["inputs.denoise"] = IMG2IMG_DENOISE.get(role, 0.60)

        if uploaded_face and not is_detail_slide(role):
            overrides["_claude_inject_ipadapter_image"] = {"inputs.image": uploaded_face}

        if poses_dir and role in POSE_FILENAMES and not is_detail_slide(role):
            pose_path = Path(poses_dir) / POSE_FILENAMES[role]
            if pose_path.exists():
                uploaded_pose = client.upload_image(str(pose_path))
                overrides["_claude_inject_controlnet_image"] = {"inputs.image": uploaded_pose}
                overrides["_claude_inject_controlnet"] = {"inputs.control_net_name": cfg["models"]["controlnet_openpose"]}

        patched = inject_workflow_values(workflow_data, overrides)
        mode_label = "img2img" if use_img2img else "t2i"
        console.print(f"[cyan]Slide {slide_num}/{len(slide_sequence)} ({role}, {mode_label}, seed {img_seed})...[/cyan]")

        try:
            prompt_id = client.submit_workflow(patched)
            images = client.wait_for_completion(prompt_id, timeout=comfy_cfg["timeout"])
        except ComfyUIError as e:
            console.print(f"[red]Slide {slide_num} failed: {e}[/red]")
            raise SystemExit(1)

        for img_meta in images:
            img_bytes = client.download_image(
                img_meta["filename"], img_meta.get("subfolder", ""), img_meta.get("type", "output")
            )
            filename = f"slide_{slide_num}_{role}_{img_seed}.png"
            dest = out_dir / filename
            dest.write_bytes(img_bytes)
            console.print(f"[green]Saved:[/green] {dest}")
            info_lines.append(f"slide_{slide_num}: {role} — {filename} (seed {img_seed})")

            # Upload slide 1 as anchor for subsequent slides
            if i == 0 and anchor_image_path is None:
                anchor_image_path = dest
                console.print(f"[dim]Uploading slide 1 as img2img anchor...[/dim]")
                uploaded_anchor = client.upload_image(str(dest))

    info_path = out_dir / "carousel_info.txt"
    info_path.write_text("\n".join(info_lines), encoding="utf-8")
    console.print(f"\n[bold green]Carousel complete![/bold green] {len(slide_sequence)} slides -> {out_dir}")
    console.print(f"[dim]Info: {info_path}[/dim]")


if __name__ == "__main__":
    main()
