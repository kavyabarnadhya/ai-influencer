"""
CLIP cosine similarity audit for v2 seed dataset.

Computes pairwise cosine similarity matrix across all images in a directory.
Flags pairs with similarity > threshold (default 0.92) as near-duplicates.
Prints a ranked list of flagged pairs and suggests which to reject.

Requires: pip install open-clip-torch torch pillow

Usage:
    python scripts/clip_similarity_audit.py --input-dir "character/ananya/seeds_v2/training_canonical"
    python scripts/clip_similarity_audit.py --input-dir "..." --threshold 0.90
    python scripts/clip_similarity_audit.py --input-dir "..." --save-matrix
"""

import json
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()
ROOT = Path(__file__).parent.parent
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_clip():
    try:
        import open_clip
        import torch
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        return model, preprocess, device, torch
    except ImportError:
        console.print("[red]open-clip-torch not installed.[/red]")
        console.print("Install: pip install open-clip-torch torch")
        raise SystemExit(1)


def encode_images(image_paths: list[Path], model, preprocess, device, torch) -> "torch.Tensor":
    from PIL import Image
    features_list = []
    for p in image_paths:
        try:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                feats = model.encode_image(img)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            features_list.append(feats.cpu())
        except Exception as e:
            console.print(f"  [yellow]Skip {p.name}: {e}[/yellow]")
            features_list.append(None)
    return features_list


@click.command()
@click.option("--input-dir", required=True, type=click.Path(exists=True),
              help="Directory of seed images to audit")
@click.option("--threshold", default=0.92, show_default=True, type=float,
              help="Cosine similarity threshold above which pairs are flagged")
@click.option("--save-matrix", is_flag=True, help="Save full similarity matrix as JSON")
@click.option("--reject-dir", default=None,
              help="If set, move lower-scoring image from each flagged pair here (dry-run shows moves)")
@click.option("--dry-run", is_flag=True, help="Show what would be rejected, don't move files")
def main(input_dir: str, threshold: float, save_matrix: bool,
         reject_dir: str | None, dry_run: bool):
    """Audit dataset for near-duplicate images via CLIP ViT-L/14 cosine similarity."""

    import torch

    input_path = Path(input_dir)
    # Optimization: os.scandir() is significantly faster than Path.iterdir() for high-volume
    # file discovery by avoiding redundant Path object allocations and suffix checks.
    exts = tuple(e.lower() for e in SUPPORTED_EXTS)
    with os.scandir(input_path) as it:
        images = sorted([
            Path(entry.path) for entry in it
            if entry.is_file() and entry.name.lower().endswith(exts)
        ])

    if len(images) < 2:
        console.print(f"[red]Need at least 2 images, found {len(images)}[/red]")
        raise SystemExit(1)

    console.print(f"[bold]CLIP Similarity Audit[/bold]")
    console.print(f"Images: {len(images)} | Threshold: {threshold}")

    model, preprocess, device, torch = load_clip()
    console.print(f"Device: {device}")

    console.print("Encoding images...")
    features_list = encode_images(images, model, preprocess, device, torch)

    # Filter out failed encodes
    valid = [(p, f) for p, f in zip(images, features_list) if f is not None]
    if len(valid) < 2:
        console.print("[red]Too few valid images to compare.[/red]")
        raise SystemExit(1)

    valid_paths, valid_feats = zip(*valid)
    feat_matrix = torch.cat(valid_feats, dim=0)  # [N, D]

    # Pairwise cosine similarity
    sim_matrix = (feat_matrix @ feat_matrix.T).numpy()

    n = len(valid_paths)
    flagged = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim > threshold:
                flagged.append((sim, valid_paths[i], valid_paths[j]))

    flagged.sort(reverse=True)

    # Average pairwise similarity (off-diagonal)
    total_sim = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_sim += float(sim_matrix[i, j])
            count += 1
    avg_sim = total_sim / count if count > 0 else 0.0

    console.print(f"\nAverage pairwise similarity: [bold]{avg_sim:.4f}[/bold] (target < 0.85)")
    if avg_sim > 0.85:
        console.print("[yellow]WARNING: avg similarity > 0.85 — dataset may be too homogeneous[/yellow]")
    else:
        console.print("[green]OK — avg similarity within target range[/green]")

    if flagged:
        table = Table(title=f"Flagged pairs (sim > {threshold})", show_lines=True)
        table.add_column("Similarity", width=10)
        table.add_column("Image A", width=40)
        table.add_column("Image B", width=40)
        table.add_column("Reject", width=30)
        for sim, pa, pb in flagged:
            # Suggest rejecting whichever sorts later alphabetically (heuristic: keep earlier curation choice)
            reject = pb.name
            table.add_row(f"{sim:.4f}", pa.name, pb.name, f"→ reject {reject}")
        console.print(table)
    else:
        console.print(f"\n[green]No flagged pairs — all pairs below {threshold}[/green]")

    if flagged and reject_dir:
        reject_path = Path(reject_dir)
        reject_path.mkdir(parents=True, exist_ok=True)
        for sim, pa, pb in flagged:
            reject_file = pb  # reject the second of each pair
            if dry_run:
                console.print(f"[dim]Would move: {reject_file.name} → {reject_path.name}/[/dim]")
            else:
                dest = reject_path / reject_file.name
                reject_file.rename(dest)
                console.print(f"Moved: {reject_file.name} → {reject_path.name}/")

    if save_matrix:
        matrix_data = {
            "images": [str(p.name) for p in valid_paths],
            "avg_similarity": avg_sim,
            "threshold": threshold,
            "flagged_pairs": [
                {"sim": sim, "a": pa.name, "b": pb.name}
                for sim, pa, pb in flagged
            ],
            "matrix": sim_matrix.tolist(),
        }
        out_file = input_path / "clip_similarity_matrix.json"
        with open(out_file, "w") as f:
            json.dump(matrix_data, f, indent=2)
        console.print(f"\nMatrix saved: {out_file}")

    console.print(f"\n[bold]Summary:[/bold] {len(flagged)} flagged pairs | avg sim {avg_sim:.4f}")
    if flagged:
        console.print(f"[yellow]Reject flagged images and replace with diverse alternatives before training.[/yellow]")


if __name__ == "__main__":
    main()
