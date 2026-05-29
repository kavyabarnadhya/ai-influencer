"""
Auto-caption seed images for Ananya v2 LoRA training.

First pass: generates raw captions using JoyCaption (via ComfyUI node) or
Florence-2 (via transformers, local CPU/GPU). Output is one .txt per image.

After running: manually edit each .txt per the Ananya v2 caption template in
character/ananya/v2_scene_anchor_vocab.md. The auto-captions are a starting
draft — NOT training-ready without editorial review.

Caption rules (from v2 plan — VERIFY these in every edited caption):
  1. AnyV2X9 must be FIRST token
  2. OMIT: face shape, eye color/shape, nose, lips, skin tone, ethnicity, body type, age
  3. INCLUDE: shot type, focal, angle, hair style+state, outfit, jewelry, pose, expression,
              lighting, DOF, aesthetic, geographic anchor
  4. Use exact vocabulary from v2_scene_anchor_vocab.md

Modes:
  --mode florence2   (default, local, CPU OK, no ComfyUI)
  --mode joycaption  (ComfyUI JoyCaption node required)
  --mode stub        (writes empty .txt stubs for manual captioning)

Usage:
    python scripts/auto_caption.py --input-dir "character/ananya/seeds_v2/training_canonical"
    python scripts/auto_caption.py --input-dir "..." --mode stub
    python scripts/auto_caption.py --input-dir "..." --mode florence2 --overwrite
"""

import re
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console

console = Console()
ROOT = Path(__file__).parent.parent
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

TRIGGER = "AnyV2X9"

# Pre-compile regex for multi-space collapse
_RE_MULTISPACE = re.compile(r"\s{2,}")

CAPTION_TEMPLATE_REMINDER = """
-- EDIT REQUIRED --
Template:
  {shot_type}, {focal_phrase} of AnyV2X9 seen from {camera_angle} at {elevation},
  with {hair_style} {hair_state}. She is {pose_action} and expressing {emotion}.
  {lighting_phrase}, {dof_phrase}, {aesthetic_mode_phrase}, {geographic_anchor_phrase}.

OMIT: face shape, skin tone, ethnicity, eye/nose/lip description, body type, age.
See: character/ananya/v2_scene_anchor_vocab.md for exact vocabulary.
"""


def caption_florence2(image_path: Path, model: Any, processor: Any, device: str) -> str:
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    prompt_text = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=prompt_text, images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=3,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(generated_text, task=prompt_text, image_size=(img.width, img.height))
    raw_caption = parsed.get(prompt_text, "")
    return raw_caption


def build_draft_caption(raw: str) -> str:
    """Prepend trigger, add edit reminder."""
    # Strip known identity-leaking terms from raw caption
    identity_terms = [
        "indian", "south asian", "brown skin", "dark skin", "tan skin",
        "black hair", "long hair", "almond eyes", "ethnic", "beautiful woman",
        "attractive", "pretty", "gorgeous", "stunning",
        "young woman", "woman in her", "woman aged",
    ]
    cleaned = raw
    for term in identity_terms:
        cleaned = cleaned.replace(term, "").replace(term.title(), "").replace(term.upper(), "")

    # Collapse multiple spaces using pre-compiled regex
    cleaned = _RE_MULTISPACE.sub(" ", cleaned).strip().strip(",").strip()

    draft = f"{TRIGGER}, {cleaned}"
    return draft


@click.command()
@click.option("--input-dir", required=True, type=click.Path(exists=True),
              help="Directory of seed images to caption")
@click.option("--mode", default="stub", type=click.Choice(["florence2", "stub"]),
              show_default=True, help="Caption mode")
@click.option("--overwrite", is_flag=True, help="Overwrite existing .txt files")
def main(input_dir: str, mode: str, overwrite: bool):
    """Auto-generate first-pass captions for v2 seed images."""

    input_path = Path(input_dir)
    images = sorted([p for p in input_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
    if not images:
        console.print(f"[red]No images in {input_path}[/red]")
        raise SystemExit(1)

    console.print(f"[bold]Auto Caption — {mode}[/bold]")
    console.print(f"Images: {len(images)} | Overwrite: {overwrite}")
    if mode == "florence2":
        console.print("[yellow]Florence-2: loading model. May take 30-60s on CPU...[/yellow]")
    if mode == "stub":
        console.print("[dim]Stub mode: writes empty .txt with template reminder. Fill manually.[/dim]")

    # Florence-2: load model once outside the loop
    model = None
    processor = None
    device = None
    if mode == "florence2":
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForCausalLM, AutoProcessor
            model_name = "microsoft/Florence-2-base"
            console.print(f"  Loading Florence-2 ({model_name})...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
            model.eval()
        except ImportError:
            console.print("[red]Required libraries (transformers, torch, or pillow) not installed.[/red]")
            console.print("Install: pip install transformers torch pillow")
            raise SystemExit(1)
        except Exception as e:
            console.print(f"[red]Failed to load Florence-2: {e}[/red]")
            raise SystemExit(1)

    skipped = 0
    written = 0
    for img_path in images:
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists() and not overwrite:
            console.print(f"  [dim]Skip (exists): {txt_path.name}[/dim]")
            skipped += 1
            continue

        console.print(f"  {img_path.name} -> {txt_path.name}")

        if mode == "stub":
            caption = f"{TRIGGER}, [FILL IN]\n\n{CAPTION_TEMPLATE_REMINDER}"
        elif mode == "florence2":
            try:
                raw = caption_florence2(img_path, model, processor, device)
                caption = build_draft_caption(raw)
            except Exception as e:
                console.print(f"  [red]Florence-2 error: {e}[/red]")
                caption = f"{TRIGGER}, [FLORENCE2 FAILED — FILL IN]\n\n{CAPTION_TEMPLATE_REMINDER}"
        else:
            caption = f"{TRIGGER}, [FILL IN]\n\n{CAPTION_TEMPLATE_REMINDER}"

        txt_path.write_text(caption, encoding="utf-8")
        written += 1

    console.print(f"\n[bold green]Done.[/bold green] {written} written | {skipped} skipped")
    console.print("\n[bold]REQUIRED next step — manual caption edit:[/bold]")
    console.print("  For every .txt in the dataset:")
    console.print("  1. Ensure AnyV2X9 is FIRST token")
    console.print("  2. Add: shot type, focal, angle, hair style+state, outfit, jewelry, pose, expression")
    console.print("  3. Add: lighting, DOF, aesthetic, geographic anchor (from v2_scene_anchor_vocab.md)")
    console.print("  4. REMOVE: any face/skin/ethnicity/body type/age description")
    console.print("  5. Verify trigger present in every caption")
    console.print("\n  Reference: character/ananya/v2_scene_anchor_vocab.md")


if __name__ == "__main__":
    main()
