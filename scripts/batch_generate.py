import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

sys.path.insert(0, str(Path(__file__).parent))
from comfyui_api import ComfyUIClient, ComfyUIError, inject_workflow_values, load_workflow, find_comfyui_port
from prompt_assistant import polish_prompt, ollama_running

console = Console()
ROOT = Path(__file__).parent.parent

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

HAND_POSITION_TAGS = ("hand", "hands", "arm", "arms", "pocket", "pockets", "strap", "crossed")

POSE_POOLS = {
    "standing": [
        "standing_01.png", "standing_02.png", "standing_03.png", "standing_04.png",
        "standing_05.png", "standing_06.png", "standing_07.png", "standing_08.png",
    ],
    "sitting": ["sitting_01.png", "sitting_02.png", "sitting_03.png", "sitting_05.png"],
    "dance":   ["dance_01.png", "dance_02.png"],
}


def pick_jewelry() -> str:
    return f"{random.choice(JEWELRY_POOL)}, {random.choice(RING_BRACELET_POOL)}"


def ensure_hand_position(prompt: str) -> str:
    p = prompt.lower()
    if any(tag in p for tag in HAND_POSITION_TAGS):
        return prompt
    seated = ("seated", "sitting", "cafe", "coffee", "table", "restaurant")
    tag = "both hands out of frame below table" if any(w in p for w in seated) else "hands not visible"
    return f"{prompt}, {tag}"


def detect_shot_type(prompt: str) -> str:
    p = prompt.lower()
    if any(w in p for w in ("close-up", "close up", "portrait", "face")):
        return "closeup"
    if any(w in p for w in ("full body", "full-body", "full length", "full-length")):
        return "fullbody"
    return "waistup"


def pick_pose(prompt: str, poses_dir: Path) -> Path | None:
    """Returns None for close-up shots (no body visible). Otherwise picks a contextually matching pose."""
    if detect_shot_type(prompt) == "closeup":
        return None
    p = prompt.lower()
    if any(w in p for w in ("dance", "dancing", "movement")):
        pool = POSE_POOLS["dance"]
    elif any(w in p for w in ("seated", "sitting", "cafe", "restaurant", "table", "hotel room", "bed", "coffee")):
        pool = POSE_POOLS["sitting"]
    else:
        pool = POSE_POOLS["standing"]
    chosen = random.choice(pool)
    path = poses_dir / chosen
    return path if path.exists() else None


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


def load_prompts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_{timestamp}.log"
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger("batch")


@click.command()
@click.option("--prompts", required=True, type=click.Path(exists=True, path_type=Path), help="Path to prompts file")
@click.option("--count-per-prompt", default=1, show_default=True, help="Images to generate per prompt")
@click.option("--category", type=click.Choice(["lifestyle", "premium"]), default="lifestyle", show_default=True)
@click.option("--character", default="ananya", show_default=True, help="Character to generate [ananya|kavib]")
@click.option("--workflow", default=None, help="Workflow override (bypasses auto pose-detection logic)")
def main(prompts: Path, count_per_prompt: int, category: str, character: str, workflow: str | None):
    cfg = load_config()
    char_cfg = load_character(cfg, character)
    comfy_cfg = cfg["comfyui"]
    gen_cfg = cfg["generation"]

    host = comfy_cfg["host"]
    port = find_comfyui_port(host, [comfy_cfg["port"], 8000, 8188, 8002])
    if port is None:
        console.print("[red]ComfyUI is not running. Start it first.[/red]")
        raise SystemExit(1)
    if port != comfy_cfg["port"]:
        console.print(f"[yellow]ComfyUI found on port {port} (config says {comfy_cfg['port']})[/yellow]")
    client = ComfyUIClient(host, port)

    use_ollama = ollama_running()
    if use_ollama:
        console.print("[dim]Ollama detected — prompts will be polished before generation.[/dim]")
    else:
        console.print("[yellow]Ollama not running — using raw prompts (quality will be lower).[/yellow]")

    prompt_list = load_prompts(prompts)
    if not prompt_list:
        console.print("[red]No prompts found in file.[/red]")
        raise SystemExit(1)

    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = ROOT / cfg["paths"]["output_dir"] / today / char_cfg["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logger(ROOT / cfg["paths"]["log_dir"])
    log.info(
        "Batch start: character=%s category=%s %d prompts x %d = %d images",
        character, category, len(prompt_list), count_per_prompt, len(prompt_list) * count_per_prompt,
    )

    trigger = char_cfg["trigger_word"]
    base_prompt = (ROOT / char_cfg["base_prompt_file"]).read_text(encoding="utf-8").strip()
    clean_base = base_prompt.replace(trigger, "").strip().strip(",")
    poses_dir = ROOT / "character" / character / "poses"

    total = len(prompt_list) * count_per_prompt
    done = 0

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("Generating...", total=total)

        for scene_prompt in prompt_list:
            if use_ollama:
                polished = ensure_hand_position(polish_prompt(scene_prompt))
            else:
                polished = ensure_hand_position(scene_prompt)
            jewelry = pick_jewelry()
            full_prompt = f"{trigger}, {polished}, {jewelry}, {clean_base}"

            # Determine workflow and pose per scene
            if workflow:
                scene_workflow = workflow
                pose_path = None
            else:
                pose_path = pick_pose(scene_prompt, poses_dir)
                scene_workflow = "t2i_sdxl_controlnet_upscale" if pose_path else "t2i_sdxl_upscale"

            if pose_path:
                console.print(f"[dim]Pose: {pose_path.name} ({detect_shot_type(scene_prompt)})[/dim]")
            else:
                console.print(f"[dim]No pose (close-up or override)[/dim]")

            workflow_path = ROOT / cfg["paths"]["workflows_dir"] / f"{scene_workflow}.json"
            if not workflow_path.exists():
                console.print(f"[red]Workflow not found: {workflow_path}[/red]")
                raise SystemExit(1)
            workflow_data = load_workflow(str(workflow_path))

            # Upload pose image once per scene (reused across count_per_prompt variations)
            uploaded_pose_name = None
            if pose_path:
                try:
                    uploaded_pose_name = client.upload_image(str(pose_path))
                except ComfyUIError as e:
                    console.print(f"[yellow]Pose upload failed, skipping pose: {e}[/yellow]")
                    log.warning("Pose upload failed for %s: %s", pose_path.name, e)

            # Performance Optimization: Pre-patch constant workflow values for this scene
            # (once per scene instead of for every image variation) to reduce overhead.
            base_overrides = {
                "_claude_inject_prompt": {"inputs.text": full_prompt},
                "_claude_inject_negative": {"inputs.text": gen_cfg["negative_prompt"]},
                "_claude_inject_latent": {"inputs.width": gen_cfg["width"], "inputs.height": gen_cfg["height"]},
                "_claude_inject_checkpoint": {"inputs.ckpt_name": cfg["models"]["checkpoint"]},
                "_claude_inject_lora": {
                    "inputs.lora_name": char_cfg["lora"],
                    "inputs.strength_model": char_cfg["lora_strength"],
                    "inputs.strength_clip": char_cfg["lora_strength"],
                },
                "_claude_inject_seed": {
                    "inputs.steps": gen_cfg["steps"],
                    "inputs.cfg": gen_cfg["cfg"]
                },
            }

            if scene_workflow in ("t2i_sdxl_upscale", "t2i_sdxl_controlnet_upscale"):
                base_overrides["_claude_inject_upscaler"] = {"inputs.model_name": cfg["models"]["upscaler"]}

            if uploaded_pose_name:
                base_overrides["_claude_inject_controlnet"] = {"inputs.control_net_name": cfg["models"]["controlnet_openpose"]}
                base_overrides["_claude_inject_controlnet_image"] = {"inputs.image": uploaded_pose_name}

            workflow_data = inject_workflow_values(workflow_data, base_overrides)

            # Queuing Optimization: Submit all variations for this scene to the ComfyUI queue first.
            # This hides download/polling latency and lets the GPU work back-to-back.
            pending_prompts: list[tuple[str, int]] = []
            for _ in range(count_per_prompt):
                seed = random.randint(0, 2**32 - 1)

                # Seed is the only value that changes per variation.
                # Other scene-level values were already patched into workflow_data outside this loop.
                overrides = {
                    "_claude_inject_seed": {"inputs.seed": seed},
                }

                # Performance Optimization: Skip cache propagation on the final injection
                # to avoid an extra dictionary copy in client.submit_workflow().
                patched = inject_workflow_values(workflow_data, overrides, propagate_cache=False)

                try:
                    prompt_id = client.submit_workflow(patched)
                    pending_prompts.append((prompt_id, seed))
                except ComfyUIError as e:
                    log.error("Failed submission seed=%d prompt=%s: %s", seed, scene_prompt[:60], e)
                    console.print(f"[yellow]Submission failed: {e}[/yellow]")
                    progress.advance(task)

            # Wait for and download results for all queued variations.
            for prompt_id, seed in pending_prompts:
                try:
                    images = client.wait_for_completion(prompt_id, timeout=comfy_cfg["timeout"])
                    for img_meta in images:
                        img_bytes = client.download_image(
                            img_meta["filename"], img_meta.get("subfolder", ""), img_meta.get("type", "output")
                        )
                        timestamp = datetime.now().strftime("%H%M%S")
                        filename = f"{character}_{today}_{timestamp}_{seed}.png"
                        dest = out_dir / filename
                        dest.write_bytes(img_bytes)
                        log.info("Saved %s (seed=%d, prompt=%s)", dest.name, seed, scene_prompt[:60])
                    done += 1
                except ComfyUIError as e:
                    log.error("Failed completion seed=%d prompt=%s: %s", seed, scene_prompt[:60], e)
                    console.print(f"[yellow]Completion failed: {e}[/yellow]")

                progress.advance(task)

    console.print(f"[green]Done: {done}/{total} images saved to {out_dir}[/green]")


if __name__ == "__main__":
    main()
