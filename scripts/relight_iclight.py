"""
IC-Light relighting pass on faceswap results.

After faceswap, face lighting often mismatches body lighting.
IC-Light harmonizes face/body by re-rendering under consistent lighting.

Requires: IC-Light ComfyUI nodes (ComfyUI-IC-Light extension).
Workflow: workflows/iclight_relight.json

Input: directory of faceswap output images (or any images needing relight).
Output: seeds_v2/experimental/relight_YYYY-MM-DD/

After this: run texture_integrity_check.py, then promote keepers to training_canonical/.

Usage:
    python scripts/relight_iclight.py --input-dir "character/ananya/seeds_v2/experimental/faceswap_raw_2026-05-08"
    python scripts/relight_iclight.py --input-dir "..." --light-mode auto
    python scripts/relight_iclight.py --input-dir "..." --dry-run
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
WORKFLOW_NAME = "iclight_relight"

# IC-Light lighting modes — map to prompt injected into workflow
LIGHT_MODES = {
    # mode_name → lighting description injected into IC-Light workflow
    "auto":        "",
    "warm_indoor": "warm tungsten indoor light, soft ambient, no hard shadows",
    "golden_hour": "golden hour outdoor light, warm amber tones, soft directional from side",
    "overcast":    "flat overcast daylight, soft even light, no shadows",
    "studio":      "softbox studio light, even illumination, neutral grey ambient",
    "rim_backlit": "rim light from behind, hair illuminated, warm backlight, studio setup",
}


def _inject_iclight(wf: dict, image_name: str, light_prompt: str) -> dict:
    """
    Inject target image and lighting prompt into IC-Light workflow.
    Optimization: Uses inject_workflow_values for O(1) node lookup and optimized copying.
    """
    overrides = {
        "_claude_inject_target_image": {"inputs.image": image_name}
    }
    if light_prompt:
        overrides["_claude_inject_light_prompt"] = {"inputs.text": light_prompt}
    return inject_workflow_values(wf, overrides)


@click.command()
@click.option("--input-dir", required=True, type=click.Path(exists=True),
              help="Directory of images to relight")
@click.option("--light-mode", default="auto", type=click.Choice(list(LIGHT_MODES.keys())),
              show_default=True, help="Lighting preset to apply")
@click.option("--workflow", default=WORKFLOW_NAME, show_default=True)
@click.option("--dry-run", is_flag=True)
@click.option("--limit", default=None, type=int, help="Process only first N images")
def main(input_dir: str, light_mode: str, workflow: str, dry_run: bool, limit: int | None):
    """IC-Light relighting batch pass for faceswap output harmonization."""

    input_path = Path(input_dir)
    images = sorted([p for p in input_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
    if not images:
        console.print(f"[red]No images in {input_path}[/red]")
        raise SystemExit(1)

    if limit:
        images = images[:limit]

    light_prompt = LIGHT_MODES[light_mode]

    console.print(f"[bold]Ananya v2 — IC-Light Relight[/bold]")
    console.print(f"Input: {len(images)} images from {input_path.name}")
    console.print(f"Light mode: {light_mode}" + (f" → '{light_prompt}'" if light_prompt else " (workflow default)"))

    if dry_run:
        for img in images:
            console.print(f"  [dim]→ {img.name}[/dim]")
        console.print(f"\n[yellow]Dry run — no images processed[/yellow]")
        return

    workflow_path = ROOT / "workflows" / f"{workflow}.json"
    if not workflow_path.exists():
        console.print(f"[red]Workflow not found: {workflow_path}[/red]")
        console.print("[yellow]Create workflows/iclight_relight.json with IC-Light nodes.[/yellow]")
        console.print("Required sentinel node titles:")
        console.print("  _claude_inject_target_image  (LoadImage)")
        console.print("  _claude_inject_light_prompt  (CLIPTextEncode, optional)")
        raise SystemExit(1)

    port = find_comfyui_port()
    if not port:
        console.print("[red]ComfyUI not found.[/red]")
        raise SystemExit(1)
    console.print(f"ComfyUI found on port {port}")

    client = ComfyUIClient(host="127.0.0.1", port=port)

    # Optimization: Load workflow template once outside the loop.
    # inject_workflow_values() returns a shallow copy, so the base template is safe.
    wf_template = load_workflow(str(workflow_path))

    date_str = datetime.now().strftime("%Y-%m-%d")
    out_path = ROOT / "character" / "ananya" / "seeds_v2" / "experimental" / f"relight_{date_str}"
    out_path.mkdir(parents=True, exist_ok=True)

    failed = []
    for i, img_path in enumerate(images, 1):
        console.print(f"\n[dim]{i}/{len(images)} — {img_path.name}[/dim]")

        try:
            uploaded = client.upload_image(str(img_path))
            wf = _inject_iclight(wf_template, image_name=uploaded, light_prompt=light_prompt)

            prompt_id = client.submit_workflow(wf)
            image_refs = client.wait_for_completion(prompt_id, timeout=120)
            if not image_refs:
                console.print(f"  [yellow]No output[/yellow]")
                failed.append(img_path.name)
                continue

            out_file = out_path / f"relit_{i:03d}_{img_path.stem}.png"
            img_bytes = client.download_image(
                image_refs[0]["filename"],
                image_refs[0].get("subfolder", ""),
                image_refs[0].get("type", "output"),
            )
            with open(out_file, "wb") as f:
                f.write(img_bytes)
            console.print(f"  Saved: {out_file.name}")

        except ComfyUIError as e:
            console.print(f"  [red]Error: {e}[/red]")
            failed.append(img_path.name)

    console.print(f"\n[bold green]Done.[/bold green] {len(images) - len(failed)}/{len(images)} relit.")
    if failed:
        console.print(f"[yellow]Failed: {failed}[/yellow]")
    console.print(f"Output: {out_path}")
    console.print("\n[bold]Next:[/bold] run texture_integrity_check.py on output dir")


if __name__ == "__main__":
    main()
