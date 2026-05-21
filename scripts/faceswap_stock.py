"""
Batch faceswap: paste Ananya v1 face onto face-obscured stock images.

Requires ReActor ComfyUI node. Calls ComfyUI with a faceswap workflow.
Source face: a clean v1 LoRA closeup (provided via --face-ref).
Target images: face-obscured stock in seeds_v2_stock_source/.

Output → seeds_v2/experimental/faceswap_raw/

After running: visually curate keepers, run relight_iclight.py on them,
then texture_integrity_check.py before promoting to training_canonical/.

Usage:
    python scripts/faceswap_stock.py --face-ref "path/to/face_ref.png"
    python scripts/faceswap_stock.py --face-ref "path/to/face_ref.png" --input-dir "character/ananya/seeds_v2_stock_source"
    python scripts/faceswap_stock.py --face-ref "path/to/face_ref.png" --files "img1.jpg,img2.jpg"
    python scripts/faceswap_stock.py --face-ref "path/to/face_ref.png" --dry-run
"""

import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from comfyui_api import ComfyUIClient, ComfyUIError, find_comfyui_port, load_workflow, inject_workflow_values

console = Console()
ROOT = Path(__file__).parent.parent

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ReActor faceswap workflow — uses _claude_inject_* sentinels:
#   _claude_inject_source_image  → face reference (Ananya closeup)
#   _claude_inject_target_image  → stock image to swap into
#   _claude_inject_output_path   → for naming only; ComfyUI saves internally
WORKFLOW_NAME = "faceswap_reactor"


def _inject_faceswap(wf: dict, face_ref_name: str, target_name: str, propagate_cache: bool = True) -> dict:
    """
    Inject face ref + target into ReActor workflow nodes by sentinel title.
    Optimization: Uses inject_workflow_values for O(1) node lookup and optimized copying.
    """
    overrides = {
        "_claude_inject_source_image": {"inputs.image": face_ref_name},
        "_claude_inject_target_image": {"inputs.image": target_name}
    }
    return inject_workflow_values(wf, overrides, propagate_cache=propagate_cache)


@click.command()
@click.option("--face-ref", required=True, type=click.Path(exists=True),
              help="Path to Ananya v1 closeup face reference image")
@click.option("--input-dir", default="character/ananya/seeds_v2_stock_source",
              show_default=True, help="Directory containing face-obscured stock images")
@click.option("--workflow", default=WORKFLOW_NAME, show_default=True)
@click.option("--dry-run", is_flag=True, help="List images that would be processed, no ComfyUI calls")
@click.option("--limit", default=None, type=int, help="Process only first N images (for testing)")
@click.option("--files", default=None, help="Comma-separated filenames to process (subset of --input-dir). If omitted, process all.")
def main(face_ref: str, input_dir: str, workflow: str, dry_run: bool, limit: int | None, files: str | None):
    """Batch ReActor faceswap: Ananya face onto stock images."""

    face_ref_path = Path(face_ref)
    input_path = ROOT / input_dir
    if not input_path.exists():
        console.print(f"[red]Input dir not found: {input_path}[/red]")
        raise SystemExit(1)

    stock_images = sorted([p for p in input_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])

    if files:
        allowed = {f.strip() for f in files.split(",")}
        stock_images = [p for p in stock_images if p.name in allowed]
        if not stock_images:
            console.print(f"[red]None of the --files entries found in {input_path}[/red]")
            raise SystemExit(1)

    if not stock_images:
        console.print(f"[red]No images found in {input_path}[/red]")
        raise SystemExit(1)

    if limit:
        stock_images = stock_images[:limit]

    console.print(f"[bold]Ananya v2 — Faceswap Stock[/bold]")
    console.print(f"Face ref: {face_ref_path.name}")
    console.print(f"Stock images: {len(stock_images)} in {input_path.name}")

    if dry_run:
        for img in stock_images:
            console.print(f"  [dim]-> {img.name}[/dim]")
        console.print(f"\n[yellow]Dry run — no images processed[/yellow]")
        return

    workflow_path = ROOT / "workflows" / f"{workflow}.json"
    if not workflow_path.exists():
        console.print(f"[red]Workflow not found: {workflow_path}[/red]")
        console.print("[yellow]Create workflows/faceswap_reactor.json with ReActor nodes.[/yellow]")
        console.print("Required sentinel node titles:")
        console.print("  _claude_inject_source_image  (LoadImage, face reference)")
        console.print("  _claude_inject_target_image  (LoadImage, stock target)")
        raise SystemExit(1)

    port = find_comfyui_port()
    if not port:
        console.print("[red]ComfyUI not found. Start ComfyUI Desktop first.[/red]")
        raise SystemExit(1)
    console.print(f"ComfyUI found on port {port}")

    client = ComfyUIClient(host="127.0.0.1", port=port)

    date_str = datetime.now().strftime("%Y-%m-%d")
    out_path = ROOT / "character" / "ananya" / "seeds_v2" / "experimental" / f"faceswap_raw_{date_str}"
    out_path.mkdir(parents=True, exist_ok=True)

    # Upload face reference once
    console.print(f"Uploading face reference: {face_ref_path.name}...")
    uploaded_face = client.upload_image(str(face_ref_path))

    # Optimization: Load workflow template once outside the loop.
    # inject_workflow_values() returns a shallow copy, so the base template is safe.
    wf_template = load_workflow(str(workflow_path))

    pending = []
    failed = []

    # Stage 1: Submit all jobs to the ComfyUI queue
    for i, stock_img in enumerate(stock_images, 1):
        console.print(f"\n[dim]Submitting {i}/{len(stock_images)} — {stock_img.name}[/dim]")
        try:
            uploaded_target = client.upload_image(str(stock_img))
            # Optimization: Skip cache propagation on final injection to avoid extra dict copy in submit_workflow
            wf = _inject_faceswap(wf_template, face_ref_name=uploaded_face, target_name=uploaded_target, propagate_cache=False)

            prompt_id = client.submit_workflow(wf)
            out_file = out_path / f"swap_{i:03d}_{stock_img.stem}.png"
            pending.append((prompt_id, stock_img, out_file))
        except ComfyUIError as e:
            console.print(f"  [red]Submission failed: {e}[/red]")
            failed.append(stock_img.name)

    # Stage 2: Wait for completion and download results
    for prompt_id, stock_img, out_file in pending:
        console.print(f"\n[dim]Processing {stock_img.name} (prompt {prompt_id})...[/dim]")
        try:
            image_refs = client.wait_for_completion(prompt_id, timeout=120)
            if not image_refs:
                console.print(f"  [yellow]No output[/yellow]")
                failed.append(stock_img.name)
                continue

            img_bytes = client.download_image(
                image_refs[0]["filename"],
                image_refs[0].get("subfolder", ""),
                image_refs[0].get("type", "output"),
            )
            with open(out_file, "wb") as f:
                f.write(img_bytes)
            console.print(f"  Saved: {out_file.name}")

        except ComfyUIError as e:
            console.print(f"  [red]Generation failed: {e}[/red]")
            failed.append(stock_img.name)

    console.print(f"\n[bold green]Done.[/bold green] {len(stock_images) - len(failed)}/{len(stock_images)} swapped.")
    if failed:
        console.print(f"[yellow]Failed: {failed}[/yellow]")
    console.print(f"Output: {out_path}")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Curate keepers visually (reject: bad face placement, wrong lighting)")
    console.print("  2. Run relight_iclight.py on keepers")
    console.print("  3. Run texture_integrity_check.py")
    console.print("  4. Promote final 15-18 to seeds_v2/training_canonical/")


if __name__ == "__main__":
    main()
