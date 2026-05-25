"""
Reprocess existing carousel base renders through ReActor + hand detail + (patched) skin lock.

Use when post-process pipeline has been fixed and you want to re-apply it to existing
FLUX renders without re-running the expensive FLUX kontext stage.

Reads `_intermediate/slide_*_base.png` (post-FLUX, pre-ReActor) -> writes `slide_*.png`
overwriting current finals.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import click
from PIL import Image
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from comfyui_api import ComfyUIClient, find_comfyui_port, load_workflow
from faceswap_carousel import _inject_faceswap, _inject_hand_detail, _run_and_save
from skin_color_match import match_body_skin_to_face_ref

console = Console()


@click.command()
@click.option("--carousel-dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="output/YYYY-MM-DD/ananya/carousel_*/ folder containing _intermediate/")
@click.option("--face-ref", default="character/ananya/seeds_v2/face_ref_v2.png",
              type=click.Path(exists=True, dir_okay=False))
@click.option("--cand", default=0, type=int, help="Candidate index to reprocess (default 0)")
def main(carousel_dir: str, face_ref: str, cand: int) -> None:
    out_dir = Path(carousel_dir).resolve()
    inter = out_dir / "_intermediate"
    if not inter.exists():
        raise click.ClickException(f"No _intermediate/ in {out_dir}")

    face_ref_path = (ROOT / face_ref).resolve()
    bases = sorted(inter.glob(f"slide_*_cand_{cand}_base.png"))
    if not bases:
        raise click.ClickException(f"No slide_*_cand_{cand}_base.png in {inter}")

    console.print(f"[cyan]Reprocessing {len(bases)} slides from {inter}[/cyan]")
    console.print(f"[cyan]face_ref: {face_ref_path}[/cyan]")

    port = find_comfyui_port()
    client = ComfyUIClient(port=port)
    faceswap_tpl = load_workflow(str(ROOT / "workflows" / "faceswap_reactor.json"))
    hand_tpl = load_workflow(str(ROOT / "workflows" / "flux_hand_detail.json"))

    uploaded_face = client.upload_image(str(face_ref_path))

    for base_path in bases:
        slide_id = base_path.stem.replace("_base", "")  # slide_00_cand_0
        final_path = out_dir / f"{slide_id}.png"
        console.print(f"\n[bold]{slide_id}[/bold] -> {final_path.name}")

        try:
            # Stage 3: ReActor
            uploaded_target = client.upload_image(str(base_path))
            wf3 = _inject_faceswap(faceswap_tpl, uploaded_face, uploaded_target,
                                   propagate_cache=False)
            _run_and_save(client, wf3, final_path, timeout=180)
            console.print(f"  ReActor OK")

            # Stage 3.5: hand detail
            try:
                uploaded_for_hands = client.upload_image(str(final_path))
                wf_hands = _inject_hand_detail(hand_tpl, uploaded_for_hands,
                                               seed=random.randint(1, 2**31 - 1),
                                               propagate_cache=False)
                _run_and_save(client, wf_hands, final_path, timeout=180)
                console.print(f"  hand_detail OK")
            except Exception as e:
                console.print(f"  [yellow]hand_detail failed: {e} — keeping ReActor output[/yellow]")

            # Stage 3.6: skin lock (patched feather)
            try:
                match_body_skin_to_face_ref(final_path, face_ref_path, final_path)
            except Exception as e:
                console.print(f"  [yellow]skin_color_match failed: {e}[/yellow]")

            # Resize to 1080×1920
            img = Image.open(final_path)
            img.resize((1080, 1920), Image.LANCZOS).save(final_path)
            console.print(f"  resized -> 1080x1920")

        except Exception as e:
            console.print(f"  [red]FAILED: {e}[/red]")

    console.print(f"\n[green]Done.[/green]")


if __name__ == "__main__":
    main()
