"""Automated cross-slide accessory/jacket continuity QA gate.

Compares a fixed crop region (the bag/torso area) across a carousel's slides
against a canonical reference slide, using CLIP embedding cosine similarity.
Fails the batch (non-zero exit) if any slide falls below threshold, so a
review can be blocked before human eyes look at it.

LIMITATION (stated plainly, not oversold): CLIP embedding similarity on a
region crop catches gross mismatches (wrong bag color, wrong garment
entirely, jacket missing) but is NOT fine-grained enough to verify exact
clasp geometry or precise hue matching pixel-for-pixel. It is a coarse
automated gate, not a substitute for the human full-resolution review this
pipeline's carousel_workflow.md already mandates.

No new model download required beyond what clip_similarity_audit.py already
uses (open-clip-torch, ViT-L-14, openai weights — already installed and
already downloaded to the local HF cache in this repo's environment).

Usage:
    python scripts/qa_accessory_continuity.py --dir output/2026-09-29/ananya/d3_coffee_run_outfit_math \
        --canonical slide_02.png --compare slide_03.png,slide_04.png \
        --crop 0.25,0.55,0.85,0.95 --threshold 0.75
"""
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


def load_clip():
    import open_clip
    import torch
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, preprocess, device, torch


def crop_region(img, box):
    """box = (x0, y0, x1, y1) as fractions of width/height."""
    w, h = img.size
    x0, y0, x1, y1 = box
    return img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))


@click.command()
@click.option("--dir", "carousel_dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--canonical", required=True, help="Filename of the canonical/approved reference slide.")
@click.option("--compare", required=True, help="Comma-separated filenames to check against the canonical.")
@click.option("--crop", default="0.15,0.55,0.9,0.95", show_default=True,
              help="x0,y0,x1,y1 as fractions of image size — the region to compare (default: lower-torso/bag band).")
@click.option("--threshold", default=0.75, type=float, show_default=True,
              help="Minimum cosine similarity to pass. Below this fails the gate.")
def main(carousel_dir: Path, canonical: str, compare: str, crop: str, threshold: float):
    from PIL import Image
    model, preprocess, device, torch = load_clip()

    box = tuple(float(x) for x in crop.split(","))
    canon_path = carousel_dir / canonical
    if not canon_path.exists():
        raise click.ClickException(f"Canonical slide not found: {canon_path}")

    with Image.open(canon_path) as img:
        canon_crop = crop_region(img.convert("RGB"), box)
        canon_tensor = preprocess(canon_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            canon_feat = model.encode_image(canon_tensor)
            canon_feat /= canon_feat.norm(dim=-1, keepdim=True)

    table = Table(title=f"Accessory continuity vs {canonical} (crop={crop}, threshold={threshold})")
    table.add_column("Slide")
    table.add_column("Cosine sim")
    table.add_column("Result")

    failures = []
    for name in compare.split(","):
        name = name.strip()
        path = carousel_dir / name
        if not path.exists():
            console.print(f"[red]Missing: {path}[/red]")
            failures.append(name)
            continue
        with Image.open(path) as img:
            c = crop_region(img.convert("RGB"), box)
            t = preprocess(c).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(t)
                feat /= feat.norm(dim=-1, keepdim=True)
            sim = float((canon_feat @ feat.T).item())
        passed = sim >= threshold
        if not passed:
            failures.append(name)
        table.add_row(name, f"{sim:.4f}", "[green]PASS[/green]" if passed else "[red]FAIL[/red]")

    console.print(table)

    if failures:
        console.print(f"\n[red]GATE FAILED[/red] — {len(failures)} slide(s) below threshold: {', '.join(failures)}")
        console.print("[yellow]Note: this crop-similarity check is coarse (whole-region CLIP embedding). "
                       "It catches gross color/garment mismatches, not precise clasp geometry. "
                       "A pass here does not replace full-resolution human review.[/yellow]")
        sys.exit(1)
    console.print("\n[green]GATE PASSED[/green] — all compared slides within threshold of the canonical crop.")
    console.print("[yellow]This is a coarse automated check, not a substitute for human full-resolution review.[/yellow]")


if __name__ == "__main__":
    main()
