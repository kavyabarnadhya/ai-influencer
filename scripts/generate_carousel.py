import random
import sys
from datetime import datetime
from pathlib import Path

import click
import yaml
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from comfyui_api import ComfyUIClient, ComfyUIError, find_comfyui_port, inject_workflow_values, load_workflow
from presets import load_preset

console = Console()
ROOT = Path(__file__).parent.parent

SLIDE_ROLES = {
    "wide": "full body shot, wide angle, sharp background, environmental storytelling, soft smile, facing camera, weight on one leg",
    "medium": "medium shot, three quarter view, body turned slightly, looking over shoulder, different pose from wide shot",
    "close": "close up portrait, bust shot, different head angle, soft smile, sharp focus on face, direct eye contact",
    "hands": "close up of hands, fabric texture detail, jewelry detail, no face visible, sharp focus",
    "ambient": "no person, background only, ambient shot, interior detail, sharp focus, no blur",
}

DEFAULT_SLIDE_SEQUENCE = ["wide", "medium", "close"]
FOUR_SLIDE_SEQUENCE = ["wide", "medium", "close", "ambient"]
POSE_ROLES = ("wide", "medium", "close")

DEFAULT_IMG2IMG_DENOISE = 0.62
DEFAULT_CONTROLNET_STRENGTH = 0.5

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


def pick_pose_for_role(poses_dir: Path, role: str, candidate_index: int = 0) -> Path | None:
    """Select a role pose deterministically by candidate index."""
    candidates = sorted(poses_dir.glob(f"{role}_*.png"))
    if not candidates:
        fallback = poses_dir / f"{role}.png"
        return fallback if fallback.exists() else None
    return candidates[candidate_index % len(candidates)]


def role_setting(carousel_cfg: dict, map_key: str, fallback_key: str, role: str, default: float) -> float:
    role_map = carousel_cfg.get(map_key, {})
    if role in role_map:
        return role_map[role]
    return carousel_cfg.get(fallback_key, default)


def build_slide_prompt(
    scene: str,
    role: str,
    char_cfg: dict,
    base_prompt: str,
    carousel_cfg: dict,
    outfit: str = "",
    hair: str = "",
    jewelry: str = "",
) -> str:
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
    body_slots = slide_count - 2
    body = (["medium", "close"] * ((body_slots + 1) // 2))[:body_slots]
    return ["wide"] + body + ["ambient"]


@click.command()
@click.option("--preset", default=None, help="Named carousel preset from character/<char>/presets.yaml")
@click.option("--scene", default=None, help="Location + lighting only, e.g. rooftop, dusk, Mumbai skyline")
@click.option("--outfit", default=None, help="Clothing locked across all model slides")
@click.option("--hair", default=None, help="Hair descriptor locked across all model slides")
@click.option("--slides", default=3, show_default=True, help="Number of carousel slides")
@click.option("--name", default=None, help="Carousel name used for output subfolder")
@click.option("--character", default="ananya", show_default=True, help="Character [ananya|kavib]")
@click.option("--seed", default=None, type=int, help="Base seed")
@click.option("--steps", default=None, type=int, help="Override step count")
@click.option("--face-ref", default=None, help="Path to face reference image for IP-Adapter")
@click.option("--poses-dir", default=None, help="Directory with role pose images, e.g. wide_01.png, medium_01.png, close_01.png")
@click.option("--candidates", default=None, type=click.IntRange(1, 5), help="Candidates to generate per model slide role")
def main(
    preset: str | None,
    scene: str | None,
    outfit: str | None,
    hair: str | None,
    slides: int,
    name: str | None,
    character: str,
    seed: int | None,
    steps: int | None,
    face_ref: str | None,
    poses_dir: str | None,
    candidates: int | None,
):
    cfg = load_config()
    char_cfg = load_character(cfg, character)
    gen_cfg = cfg["generation"]
    carousel_cfg = cfg["carousel"]
    comfy_cfg = cfg["comfyui"]

    # Resolve preset — CLI flags win over preset values
    preset_data = None
    preset_per_role = {}
    preset_overrides = {}
    if preset:
        try:
            preset_data = load_preset(character, preset)
        except (FileNotFoundError, KeyError) as e:
            console.print(f"[red]{e}[/red]")
            raise SystemExit(1)
        if preset_data.get("kind") != "carousel":
            console.print(f"[red]'{preset}' is not a carousel preset — use generate.py --preset {preset}[/red]")
            raise SystemExit(1)
        preset_per_role = preset_data.get("per_role", {})
        preset_overrides = preset_data.get("overrides", {})
        scene      = scene      or preset_data.get("scene_default", "")
        outfit     = outfit     or preset_data.get("outfit_default", "")
        hair       = hair       or preset_data.get("hair_default", "dark hair neatly styled")
        name       = name       or preset
        face_ref   = face_ref   or preset_data.get("face_ref")
        poses_dir  = poses_dir  or preset_data.get("poses_dir")
        if candidates is None:
            candidates = preset_data.get("candidates", 1)

    # Apply defaults for non-preset path
    if hair is None:
        hair = "dark hair neatly styled"
    if candidates is None:
        candidates = 1

    if not scene or not outfit or not name:
        console.print("[red]Need --scene, --outfit, --name (or --preset that supplies them)[/red]")
        raise SystemExit(1)

    host = comfy_cfg["host"]
    port = find_comfyui_port(host, [comfy_cfg["port"], 8000, 8188, 8002])
    if port is None:
        console.print("[red]ComfyUI is not running. Start it first.[/red]")
        raise SystemExit(1)
    if port != comfy_cfg["port"]:
        console.print(f"[yellow]ComfyUI found on port {port}[/yellow]")
    client = ComfyUIClient(host, port)
    client._upload_cache.clear()

    if poses_dir and face_ref:
        anchor_workflow_name = "t2i_sdxl_lora_ipadapter_controlnet"
        img2img_workflow_name = "t2i_sdxl_lora_img2img_ipadapter_controlnet"
    elif poses_dir:
        anchor_workflow_name = "t2i_sdxl_lora_controlnet"
        img2img_workflow_name = "t2i_img2img"
    elif face_ref:
        anchor_workflow_name = "t2i_sdxl_lora_ipadapter"
        img2img_workflow_name = "t2i_sdxl_lora_img2img_ipadapter"
    else:
        anchor_workflow_name = gen_cfg["workflow"]
        img2img_workflow_name = "t2i_img2img"

    close_workflow = "t2i_sdxl_lora_ipadapter_controlnet_facedetail" if face_ref else None
    workflows_to_check = set(filter(None, [anchor_workflow_name, img2img_workflow_name, "t2i_sdxl_lora", close_workflow]))
    for workflow_name in workflows_to_check:
        workflow_path = ROOT / cfg["paths"]["workflows_dir"] / f"{workflow_name}.json"
        if not workflow_path.exists():
            console.print(f"[red]Workflow not found: {workflow_path}[/red]")
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

    if preset_data and preset_data.get("slide_sequence"):
        slide_sequence = preset_data["slide_sequence"]
    else:
        slide_sequence = get_slide_sequence(slides)
    locked_jewelry = pick_jewelry()
    console.print(f"[bold green]Carousel: {name}[/bold green] - {len(slide_sequence)} slides: {slide_sequence}")
    console.print(f"[dim]Scene: {scene}[/dim]")
    console.print(f"[dim]Jewelry: {locked_jewelry}[/dim]")

    info_lines = [
        f"# Carousel: {name}",
        f"Preset: {preset or 'n/a'}",
        f"Scene: {scene}",
        f"Outfit: {outfit}",
        f"Hair: {hair}",
        f"Generated: {today}",
        f"Slides: {len(slide_sequence)}",
        f"LoRA strength: {preset_overrides.get('lora_strength') or char_cfg['lora_strength']}",
        f"IPAdapter strength: {preset_overrides.get('ipadapter_strength') or char_cfg.get('ipadapter_strength', 'n/a') if face_ref else 'n/a'}",
        f"img2img denoise by role: {carousel_cfg.get('img2img_denoise_by_role', {})}",
        f"ControlNet strength by role: {carousel_cfg.get('controlnet_strength_by_role', {})}",
        f"Poses dir: {poses_dir or 'n/a'}",
        f"Candidates per model slide: {candidates}",
        "",
    ]

    anchor_image_path: Path | None = None
    uploaded_anchor: str | None = None

    for i, role in enumerate(slide_sequence):
        slide_num = i + 1
        slide_prompt = build_slide_prompt(
            scene,
            role,
            char_cfg,
            base_prompt,
            carousel_cfg,
            outfit=outfit,
            hair=hair,
            jewelry=locked_jewelry,
        )
        role_candidate_count = candidates if not is_detail_slide(role) else 1

        for candidate_index in range(role_candidate_count):
            candidate_num = candidate_index + 1
            img_seed = base_seed + (i * 100) + candidate_index
            use_img2img = (i > 0) and not is_detail_slide(role) and anchor_image_path is not None

            role_overrides = preset_per_role.get(role, {})
            if is_detail_slide(role):
                current_workflow_name = role_overrides.get("workflow", "t2i_sdxl_lora")
            elif role_overrides.get("workflow"):
                current_workflow_name = role_overrides["workflow"]
            else:
                current_workflow_name = img2img_workflow_name if use_img2img else anchor_workflow_name
            workflow_data = load_workflow(str(ROOT / cfg["paths"]["workflows_dir"] / f"{current_workflow_name}.json"))

            denoise = None
            controlnet_strength = None
            pose_name = "n/a"

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
                denoise = role_setting(
                    carousel_cfg,
                    "img2img_denoise_by_role",
                    "img2img_denoise",
                    role,
                    DEFAULT_IMG2IMG_DENOISE,
                )
                # Preset per_role denoise wins over config.yaml
                if "denoise" in role_overrides:
                    denoise = role_overrides["denoise"]
                overrides["_claude_inject_init_image"] = {"inputs.image": uploaded_anchor}
                overrides["_claude_inject_seed"]["inputs.denoise"] = denoise

            ipadapter_weight = preset_overrides.get("ipadapter_strength") or char_cfg.get("ipadapter_strength", 0.5)
            if uploaded_face and not is_detail_slide(role):
                overrides["_claude_inject_ipadapter_image"] = {"inputs.image": uploaded_face}
                overrides["_claude_inject_ipadapter_strength"] = {"inputs.weight": ipadapter_weight}

            if poses_dir and role in POSE_ROLES and not is_detail_slide(role):
                pose_path = pick_pose_for_role(Path(poses_dir), role, candidate_index)
                if pose_path:
                    pose_name = pose_path.name
                    console.print(f"[dim]  pose: {pose_name}[/dim]")
                    uploaded_pose = client.upload_image(str(pose_path))
                    controlnet_strength = role_setting(
                        carousel_cfg,
                        "controlnet_strength_by_role",
                        "controlnet_strength",
                        role,
                        DEFAULT_CONTROLNET_STRENGTH,
                    )
                    # Preset per_role controlnet_strength wins over config.yaml
                    if "controlnet_strength" in role_overrides:
                        controlnet_strength = role_overrides["controlnet_strength"]
                    overrides["_claude_inject_controlnet_image"] = {"inputs.image": uploaded_pose}
                    overrides["_claude_inject_controlnet"] = {"inputs.control_net_name": cfg["models"]["controlnet_openpose"]}
                    overrides["ControlNet Apply"] = {"inputs.strength": controlnet_strength}

            patched = inject_workflow_values(workflow_data, overrides)
            mode_label = "img2img" if use_img2img else "t2i"
            console.print(
                f"[cyan]Slide {slide_num}/{len(slide_sequence)} cand {candidate_num}/{role_candidate_count} "
                f"({role}, {mode_label}, seed {img_seed})...[/cyan]"
            )

            try:
                prompt_id = client.submit_workflow(patched)
                images = client.wait_for_completion(prompt_id, timeout=comfy_cfg["timeout"])
            except ComfyUIError as e:
                console.print(f"[red]Slide {slide_num} candidate {candidate_num} failed: {e}[/red]")
                raise SystemExit(1)

            for img_meta in images:
                img_bytes = client.download_image(
                    img_meta["filename"], img_meta.get("subfolder", ""), img_meta.get("type", "output")
                )
                if role_candidate_count > 1:
                    filename = f"slide_{slide_num}_{role}_cand_{candidate_num}_{img_seed}.png"
                else:
                    filename = f"slide_{slide_num}_{role}_{img_seed}.png"
                dest = out_dir / filename
                dest.write_bytes(img_bytes)
                console.print(f"[green]Saved:[/green] {dest}")
                info_lines.append(
                    f"slide_{slide_num}: {role} - cand {candidate_num} - {mode_label} - "
                    f"pose {pose_name} - denoise {denoise if denoise is not None else 'n/a'} - "
                    f"controlnet {controlnet_strength if controlnet_strength is not None else 'n/a'} - "
                    f"{filename} (seed {img_seed})"
                )

                if i == 0 and candidate_index == 0 and anchor_image_path is None:
                    anchor_image_path = dest
                    console.print("[dim]Uploading slide 1 candidate 1 as img2img anchor...[/dim]")
                    uploaded_anchor = client.upload_image(str(dest))

    info_path = out_dir / "carousel_info.txt"
    info_path.write_text("\n".join(info_lines), encoding="utf-8")
    console.print(f"\n[bold green]Carousel complete![/bold green] {len(slide_sequence)} slides -> {out_dir}")
    console.print(f"[dim]Info: {info_path}[/dim]")


if __name__ == "__main__":
    main()
