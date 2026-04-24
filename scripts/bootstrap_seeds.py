#!/usr/bin/env python3
"""Generate candidate seed images for LoRA training dataset.

Runs SDXL text-only (no LoRA — LoRA doesn't exist yet) and outputs
16 candidate images per mode. You then manually pick the best 8
from each batch and copy them to the appropriate seeds/ subfolder.

Usage:
    python scripts/bootstrap_seeds.py --mode closeup
    python scripts/bootstrap_seeds.py --mode medium
    python scripts/bootstrap_seeds.py --mode fullbody
    python scripts/bootstrap_seeds.py --mode closeup --count 24

Curation guidance:
    Pick images with the sharpest face detail, most varied
    poses/lighting, and no visible SDXL artifacts (e.g., melted ears,
    extra fingers, plastic skin). Copy selected images to:
        character/seeds/closeup/   (need 8 minimum)
        character/seeds/medium/    (need 8 minimum)
        character/seeds/fullbody/  (need 8 minimum)

Dataset robustness note:
    Training a LoRA from only 24 synthetic images risks "teaching back"
    SDXL rendering artifacts instead of learning a robust identity.
    Supplement with 6-8 real-camera reference photos:
        - Own photos of a consenting adult model (explicit permission), or
        - Properly licensed stock photos of a consenting adult model.
        - NEVER celebrities, influencers, or scraped third-party faces.
    Place real photos in the seeds/ subfolders before training.
    prepare_training_data.py --validate will count them in the totals.
"""

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
    import scripts.generate as gen_module
except ImportError:
    print("Missing dependencies. Run: pip install click PyYAML rich")
    sys.exit(1)

console = Console()

# Mode-specific prompt fragments appended to the base character description.
# No KaviB trigger here — no LoRA is loaded in bootstrap_seeds.json.
MODE_PROMPTS = {
    "closeup": [
        "extreme close-up portrait, chin to forehead framing, sharp facial features, "
        "studio lighting, shallow depth of field, skin pore detail, iris detail",
        "tight face shot, head and shoulders, natural window light, clean background, "
        "photorealistic skin texture, high detail",
        "close-up portrait outdoors, dappled sunlight, short hair, looking at camera, "
        "razor-sharp focus on eyes",
        "indoor close-up portrait, soft diffused light, neutral background, "
        "emphasizing jawline and cheekbones, photorealistic",
    ],
    "medium": [
        "waist-up portrait, casual outfit, bright natural light, neutral background, "
        "standing relaxed, shoulders and arms visible",
        "half-body shot, sitting at a cafe table, window light, hands on table, "
        "lifestyle photography style",
        "medium shot, standing against a white wall, street style, "
        "arms at sides, confident pose",
        "waist-up, slight three-quarter angle, golden hour outdoor light, "
        "hair visible, casual clothing",
    ],
    "fullbody": [
        "full body portrait, head to toe, standing on a city sidewalk, "
        "full height visible, casual fashion, natural daylight",
        "full figure standing in a park, both feet visible, "
        "wide shot showing full body proportions, green background",
        "full body shot, standing in front of a minimalist white wall, "
        "fashion photography, balanced lighting",
        "full-length portrait, walking toward camera on a cobblestone street, "
        "natural light, full height and body visible",
    ],
}

NEGATIVE_PROMPT = (
    "blurry, extra fingers, malformed hands, deformed eyes, bad anatomy, "
    "duplicate person, watermark, text, low quality, cropped, "
    "out of frame, cut off"
)

BASE_CHARACTER_PROMPT = (
    "photorealistic photograph of a young woman with short hair, "
    "professional camera, hyperrealistic skin texture"
)


@click.command()
@click.option("--mode", required=True,
              type=click.Choice(["closeup", "medium", "fullbody"]),
              help="Framing mode for candidate images")
@click.option("--count", default=16, show_default=True,
              help="Number of candidate images to generate (pick best 8 to keep)")
@click.option("--config", "config_path", default="config.yaml", show_default=True)
@click.option("--seed", default=-1, show_default=True, help="Base seed (-1 for random)")
def main(mode, count, config_path, seed):
    cfg = gen_module.load_config(PROJECT_ROOT / config_path)

    console.print(
        f"\n[bold]Bootstrap Seeds — {mode.upper()} mode[/bold]\n"
        f"  Generating {count} candidates. Pick the best 8 and copy to:\n"
        f"  character/seeds/{mode}/\n"
    )
    console.print(
        "[dim]No LoRA loaded — this workflow uses base Juggernaut XL only.[/dim]\n"
    )

    out_dir = PROJECT_ROOT / "output" / "_bootstrap" / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        wf_template = gen_module.load_workflow("bootstrap_seeds")
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    mode_prompt_options = MODE_PROMPTS[mode]
    client = ComfyUIClient(host=cfg["comfyui"]["host"], port=cfg["comfyui"]["port"])
    console.print("Waiting for ComfyUI...")
    try:
        client.wait_for_ready(timeout=cfg["comfyui"]["startup_timeout"])
    except ComfyUIError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    import copy
    generated = []

    for i in range(count):
        actual_seed = (seed + i) if seed != -1 else random.randint(0, 2**32 - 1)
        # Cycle through prompt variants for diversity
        mode_frag = mode_prompt_options[i % len(mode_prompt_options)]
        full_prompt = f"{BASE_CHARACTER_PROMPT}, {mode_frag}"

        import scripts.generate as gm
        wf = copy.deepcopy(wf_template)
        for node in wf.values():
            if not isinstance(node, dict):
                continue
            title = node.get("_meta", {}).get("title", "")
            if title == "_claude_inject_prompt":
                node.setdefault("inputs", {})["text"] = full_prompt
            elif title == "_claude_inject_negative":
                node.setdefault("inputs", {})["text"] = NEGATIVE_PROMPT
            elif title == "_claude_inject_seed":
                node.setdefault("inputs", {})["seed"] = actual_seed

        console.print(f"  [{i+1}/{count}] seed={actual_seed} ...", end=" ")
        try:
            images = client.submit_and_wait(wf, str(out_dir))
            generated.extend(images)
            console.print(f"[green]OK[/green]")
        except ComfyUIError as e:
            console.print(f"[red]FAILED[/red]: {e}")

    console.print(
        f"\n[bold green]Done.[/bold green] {len(generated)} images saved to:\n"
        f"  {out_dir}\n"
        f"\n[bold]Next step:[/bold]\n"
        f"  Review the images. Pick the {8} best for your dataset.\n"
        f"  Copy selected images to:\n"
        f"    character/seeds/{mode}/\n"
        f"\n  Curation criteria:\n"
        f"    • Sharpest face detail and iris clarity\n"
        f"    • Varied poses/lighting across the 8 picks\n"
        f"    • No SDXL artifacts (melted features, extra fingers)\n"
        f"\n  Then run the next mode:\n"
        f"    python scripts/bootstrap_seeds.py --mode medium\n"
        f"\n  After all 3 modes:\n"
        f"    python scripts/prepare_training_data.py --validate"
    )


if __name__ == "__main__":
    main()
