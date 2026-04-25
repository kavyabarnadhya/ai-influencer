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
def main(prompts: Path, count_per_prompt: int, category: str, character: str):
    cfg = load_config()
    char_cfg = load_character(cfg, character)
    comfy_cfg = cfg["comfyui"]
    gen_cfg = cfg["generation"]

    client = ComfyUIClient(comfy_cfg["host"], comfy_cfg["port"])
    if not client.is_running():
        console.print("[red]ComfyUI is not running. Start it first.[/red]")
        raise SystemExit(1)

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

    base_prompt = (ROOT / char_cfg["base_prompt_file"]).read_text(encoding="utf-8").strip()
    trigger = char_cfg["trigger_word"]

    workflow_name = gen_cfg["workflow"]
    workflow_path = ROOT / cfg["paths"]["workflows_dir"] / f"{workflow_name}.json"
    workflow_data = load_workflow(str(workflow_path))

    total = len(prompt_list) * count_per_prompt
    done = 0

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("Generating...", total=total)

        for scene_prompt in prompt_list:
            full_prompt = f"{base_prompt}, {scene_prompt}"
            if not full_prompt.startswith(trigger):
                full_prompt = f"{trigger}, {full_prompt}"

            for _ in range(count_per_prompt):
                seed = random.randint(0, 2**32 - 1)

                overrides = {
                    "_claude_inject_prompt": {"inputs.text": full_prompt},
                    "_claude_inject_negative": {"inputs.text": gen_cfg["negative_prompt"]},
                    "_claude_inject_seed": {"inputs.seed": seed, "inputs.steps": gen_cfg["steps"], "inputs.cfg": gen_cfg["cfg"]},
                    "_claude_inject_latent": {"inputs.width": gen_cfg["width"], "inputs.height": gen_cfg["height"]},
                    "_claude_inject_checkpoint": {"inputs.ckpt_name": cfg["models"]["checkpoint"]},
                    "_claude_inject_lora": {
                        "inputs.lora_name": char_cfg["lora"],
                        "inputs.strength_model": char_cfg["lora_strength"],
                        "inputs.strength_clip": char_cfg["lora_strength"],
                    },
                }

                patched = inject_workflow_values(workflow_data, overrides)

                try:
                    prompt_id = client.submit_workflow(patched)
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
                    log.error("Failed seed=%d prompt=%s: %s", seed, scene_prompt[:60], e)
                    console.print(f"[yellow]Skipped (error): {e}[/yellow]")

                progress.advance(task)

    console.print(f"[green]Done: {done}/{total} images saved to {out_dir}[/green]")


if __name__ == "__main__":
    main()
