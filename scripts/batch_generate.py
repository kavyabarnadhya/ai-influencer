#!/usr/bin/env python3
"""Batch image generation from a prompt file.

Usage:
    python scripts/batch_generate.py \\
        --prompts character/prompts/lifestyle_prompts.txt \\
        --count-per-prompt 3 \\
        --category lifestyle

Adult-tier usage (requires explicit confirmation):
    python scripts/batch_generate.py \\
        --prompts character/prompts/intimate_prompts.txt \\
        --count-per-prompt 2 \\
        --category adult \\
        --adult-consent-confirmed
"""

import datetime
import json
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

try:
    import click
    import yaml
    from rich.console import Console
    from comfyui_api import ComfyUIClient, ComfyUIError
    import scripts.generate as gen_module
except ImportError:
    print("Missing dependencies. Run: pip install click PyYAML rich")
    sys.exit(1)

console = Console()

ADULT_CATEGORIES = {"adult", "intimate", "explicit", "private"}
ADULT_CHECKLIST = """
╔══════════════════════════════════════════════════════════════╗
║              ADULT CONTENT COMPLIANCE CHECKLIST              ║
╠══════════════════════════════════════════════════════════════╣
║  Before proceeding, confirm ALL of the following:           ║
║                                                              ║
║  ✓ Platform ToS: Fanvue / Telegram ONLY                     ║
║    (Instagram prohibits this content — never post there)    ║
║                                                              ║
║  ✓ AI disclosure: "#AI" or AI creator label on EVERY post   ║
║                                                              ║
║  ✓ Age representation: persona stated as 23 in ALL          ║
║    captions, bios, and profile descriptions                  ║
║                                                              ║
║  ✓ Training data: no real-person likeness in dataset         ║
║    without explicit written consent from that individual    ║
║                                                              ║
║  ✓ Outputs go to output/private/ — never mixed with         ║
║    lifestyle content                                         ║
╚══════════════════════════════════════════════════════════════╝
"""

MIN_FREE_GB = 2.0


def check_disk_space(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def load_prompts(prompts_file: Path) -> list[str]:
    lines = prompts_file.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def write_batch_log(log_path: Path, entry: dict) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


@click.command()
@click.option("--prompts", required=True, type=click.Path(exists=True),
              help="Path to prompt file (one prompt per line, # = comment)")
@click.option("--count-per-prompt", default=2, show_default=True,
              help="Images to generate per prompt")
@click.option("--category", required=True,
              help="Output category: lifestyle, portrait, adult, etc.")
@click.option("--workflow", default="t2i_sdxl_lora", show_default=True)
@click.option("--seed", default=-1, show_default=True, help="Base seed (-1 for random)")
@click.option("--steps", default=None, type=int)
@click.option("--cfg", "cfg_scale", default=None, type=float)
@click.option("--width", default=None, type=int)
@click.option("--height", default=None, type=int)
@click.option("--no-adetailer", is_flag=True)
@click.option("--adult-consent-confirmed", is_flag=True,
              help="Required for adult/intimate categories")
@click.option("--config", "config_path", default="config.yaml", show_default=True)
def main(prompts, count_per_prompt, category, workflow, seed, steps, cfg_scale,
         width, height, no_adetailer, adult_consent_confirmed, config_path):

    cfg = gen_module.load_config(PROJECT_ROOT / config_path)
    gen = cfg["generation"]

    # --- Adult compliance gate ---
    is_adult = category.lower() in ADULT_CATEGORIES
    if is_adult:
        if not adult_consent_confirmed:
            console.print(ADULT_CHECKLIST)
            console.print(
                "[red]ERROR:[/red] Adult-tier content requires --adult-consent-confirmed flag.\n"
                "Add it to your command after reading and confirming the checklist above."
            )
            sys.exit(1)
        console.print(ADULT_CHECKLIST)
        console.print("[yellow]Adult-tier mode active. Outputs → output/private/[/yellow]\n")

    steps = steps or gen["default_steps"]
    cfg_scale = cfg_scale or gen["default_cfg"]
    width = width or gen["default_width"]
    height = height or gen["default_height"]

    date_str = datetime.date.today().isoformat()
    if is_adult:
        out_dir = PROJECT_ROOT / "output" / "private" / date_str
    else:
        out_dir = PROJECT_ROOT / gen["output_dir"] / date_str / category
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = PROJECT_ROOT / cfg["logging"]["log_dir"] / f"batch_{date_str}_{category}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    prompts_list = load_prompts(Path(prompts))
    if not prompts_list:
        console.print("[red]No prompts found in file (all lines are comments or empty)[/red]")
        sys.exit(1)

    console.print(
        f"[bold]Batch Generation[/bold]\n"
        f"  Prompts      : {len(prompts_list)}\n"
        f"  Per prompt   : {count_per_prompt}\n"
        f"  Total images : {len(prompts_list) * count_per_prompt}\n"
        f"  Category     : {category}\n"
        f"  Output       : {out_dir}\n"
        f"  Log          : {log_path}\n"
    )

    try:
        wf_template = gen_module.load_workflow(workflow)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    client = ComfyUIClient(host=cfg["comfyui"]["host"], port=cfg["comfyui"]["port"])
    console.print("Waiting for ComfyUI...")
    try:
        client.wait_for_ready(timeout=cfg["comfyui"]["startup_timeout"])
    except ComfyUIError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    total = 0
    failed = 0

    for p_idx, prompt_text in enumerate(prompts_list, 1):
        # Disk space check before each prompt batch
        free_gb = check_disk_space(out_dir)
        if free_gb < MIN_FREE_GB:
            console.print(
                f"[red]Stopping:[/red] Only {free_gb:.1f}GB free disk space. "
                f"Need at least {MIN_FREE_GB}GB."
            )
            break

        console.print(f"\n[bold][{p_idx}/{len(prompts_list)}][/bold] {prompt_text[:80]}")

        for img_idx in range(count_per_prompt):
            actual_seed = (seed + p_idx * 100 + img_idx) if seed != -1 else random.randint(0, 2**32 - 1)
            injections = gen_module.build_injections(
                cfg, prompt_text, actual_seed, steps, cfg_scale, width, height, not no_adetailer
            )
            wf = gen_module.inject_into_workflow(wf_template, injections)

            console.print(f"  [{img_idx+1}/{count_per_prompt}] seed={actual_seed} ...", end=" ")
            try:
                images = client.submit_and_wait(wf, str(out_dir))
                for img_path in images:
                    entry = {
                        "prompt": prompt_text,
                        "full_prompt": injections["_claude_inject_prompt"],
                        "negative_prompt": injections["_claude_inject_negative"],
                        "category": category,
                        "workflow": workflow,
                        "seed": actual_seed,
                        "steps": steps,
                        "cfg": cfg_scale,
                        "width": width,
                        "height": height,
                        "adetailer": not no_adetailer,
                        "image": str(img_path),
                        "date": date_str,
                    }
                    gen_module.write_sidecar(img_path, entry)
                    write_batch_log(log_path, entry)
                console.print(f"[green]OK[/green]")
                total += 1
            except ComfyUIError as e:
                console.print(f"[red]FAILED[/red]: {e}")
                failed += 1

    console.print(
        f"\n[bold]Batch complete.[/bold] "
        f"Generated: {total}  Failed: {failed}\n"
        f"Output: {out_dir}\n"
        f"Log: {log_path}"
    )
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
