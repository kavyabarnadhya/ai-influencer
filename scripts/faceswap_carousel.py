"""
Faceswap-driven carousel generator for Ananya v2 (LoRA-free).

Pipeline per slide (2 stages, no compositing):
  1. FLUX schnell img2img off dark anchor body → person+outfit+scene rendered
     together natively. Anchor img2img at moderate denoise locks body proportions
     across slides while prompt fully controls outfit + scene + lighting.
  2. ReActor face swap → identity locked to face_ref_v2.png.

No background compositing, no inpaint, no rembg. FLUX renders person and BG
together in a single forward pass — perspective, scale, and lighting integrate
natively. Body consistency comes from shared anchor seed + img2img denoise.

Two modes:
  - Default (varied outfits): anchor is generic 23yo woman in dark basics.
  - --outfit-lock (OOTD): anchor wears target outfit, slides vary pose/scene only.

Prompt file format:
    denoise=0.75 | <full slide prompt>
    <full slide prompt>    # bare line uses default denoise

See character/ananya/prompt_template.md for prompt structure.

Usage:
    python scripts/faceswap_carousel.py \\
      --prompts character/ananya/carousel_prompts/smoke_test.txt \\
      --face-ref character/ananya/seeds_v2/face_ref_v2.png \\
      --name smoke_08
"""

import random
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from comfyui_api import ComfyUIClient, ComfyUIError, find_comfyui_port, load_workflow

console = Console()
ROOT = Path(__file__).parent.parent

# Anchor prompt: matches photographic aesthetic of canonical training set.
# Dark-clothed so outfit color in slides doesn't pull toward beige.
# Natural lighting + 50mm framing seeds FLUX with photographic style anchor.
DEFAULT_ANCHOR_PROMPT = (
    "photorealistic candid fashion photograph, shot on Sony A7IV 50mm, "
    "full-body three-quarter standing pose of a 23-year-old South Asian woman, "
    "dark hair loose over shoulders, plain neutral grey form-fitting bodysuit, "
    "soft natural daylight, plain light grey backdrop, "
    "natural skin texture with visible pores, light film grain, sharp focus on face, "
    "no plastic AI smoothing"
)
DEFAULT_DENOISE_VARIED = 0.85
DEFAULT_DENOISE_OUTFIT_LOCK = 0.40


def _inject_flux_t2i(wf: dict, prompt: str, seed: int) -> dict:
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        title = node.get("_meta", {}).get("title", "")
        if title == "_claude_inject_prompt":
            node["inputs"]["text"] = prompt
        elif title == "_claude_inject_seed":
            node["inputs"]["seed"] = seed
    return wf


def _inject_flux_img2img(wf: dict, prompt: str, init_image_name: str,
                        denoise: float, seed: int) -> dict:
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        title = node.get("_meta", {}).get("title", "")
        if title == "_claude_inject_prompt":
            node["inputs"]["text"] = prompt
        elif title == "_claude_inject_init_image":
            node["inputs"]["image"] = init_image_name
        elif title == "_claude_inject_seed":
            node["inputs"]["seed"] = seed
            node["inputs"]["denoise"] = denoise
    return wf


def _inject_faceswap(wf: dict, face_ref_name: str, target_name: str) -> dict:
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        title = node.get("_meta", {}).get("title", "")
        if title == "_claude_inject_source_image":
            node["inputs"]["image"] = face_ref_name
        elif title == "_claude_inject_target_image":
            node["inputs"]["image"] = target_name
    return wf


def _run_and_save(client: ComfyUIClient, wf: dict, out_path: Path,
                  timeout: int = 300) -> Path:
    prompt_id = client.submit_workflow(wf)
    image_refs = client.wait_for_completion(prompt_id, timeout=timeout)
    if not image_refs:
        raise ComfyUIError(f"No output for prompt {prompt_id}")
    img_bytes = client.download_image(
        image_refs[0]["filename"],
        image_refs[0].get("subfolder", ""),
        image_refs[0].get("type", "output"),
    )
    out_path.write_bytes(img_bytes)
    return out_path


def parse_prompts_file(path: Path, default_denoise: float) -> list[tuple[float, str]]:
    """Parse 'denoise=X | prompt' or bare 'prompt' lines."""
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        denoise = default_denoise
        text = line
        if "|" in line and line.lower().lstrip().startswith("denoise="):
            head, _, rest = line.partition("|")
            try:
                denoise = float(head.strip().split("=", 1)[1])
            except (ValueError, IndexError):
                pass
            text = rest.strip()
        out.append((denoise, text))
    return out


@click.command()
@click.option("--prompts", "prompts_file", required=True, type=click.Path(exists=True))
@click.option("--face-ref", required=True, type=click.Path(exists=True))
@click.option("--name", required=True)
@click.option("--candidates", default=1, show_default=True, type=int)
@click.option("--anchor-seed", default=None, type=int)
@click.option("--outfit-lock", is_flag=True)
@click.option("--anchor-outfit-prompt", default=None)
@click.option("--anchor-prompt", default=None)
@click.option("--character", default="ananya", show_default=True)
def main(prompts_file: str, face_ref: str, name: str, candidates: int,
         anchor_seed: int | None, outfit_lock: bool,
         anchor_outfit_prompt: str | None, anchor_prompt: str | None,
         character: str):
    """Carousel: FLUX img2img (person+BG together) → ReActor face swap."""

    prompts_path = Path(prompts_file)
    face_ref_path = Path(face_ref)

    if outfit_lock:
        if not anchor_outfit_prompt:
            console.print("[red]--outfit-lock requires --anchor-outfit-prompt[/red]")
            raise SystemExit(1)
        anchor_text = anchor_outfit_prompt
        default_denoise = DEFAULT_DENOISE_OUTFIT_LOCK
        mode_label = "outfit-lock (OOTD)"
    else:
        anchor_text = anchor_prompt or DEFAULT_ANCHOR_PROMPT
        default_denoise = DEFAULT_DENOISE_VARIED
        mode_label = "varied-outfit"

    slides = parse_prompts_file(prompts_path, default_denoise)
    if not slides:
        console.print(f"[red]No slides parsed from {prompts_path}[/red]")
        raise SystemExit(1)

    if anchor_seed is None:
        anchor_seed = random.randint(1, 2**31 - 1)

    console.print(f"[bold]Faceswap Carousel — {name}[/bold]")
    console.print(f"Mode: {mode_label} | Slides: {len(slides)} | Cands: {candidates}")
    console.print(f"Face ref: {face_ref_path.name} | Anchor seed: {anchor_seed}")
    console.print(f"Default denoise: {default_denoise}")

    port = find_comfyui_port()
    if not port:
        console.print("[red]ComfyUI not running.[/red]")
        raise SystemExit(1)
    console.print(f"ComfyUI on port {port}")
    client = ComfyUIClient(host="127.0.0.1", port=port)

    date_str = datetime.now().strftime("%Y-%m-%d")
    out_dir = ROOT / "output" / date_str / character / f"carousel_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    inter = out_dir / "_intermediate"
    inter.mkdir(exist_ok=True)

    uploaded_face = client.upload_image(str(face_ref_path))

    # Stage 1: anchor body
    console.print(f"\n[bold cyan]Stage 1 — Anchor body[/bold cyan]")
    anchor_path = inter / "anchor.png"
    wf1 = _inject_flux_t2i(
        load_workflow(str(ROOT / "workflows" / "flux_schnell.json")),
        anchor_text, anchor_seed,
    )
    try:
        _run_and_save(client, wf1, anchor_path)
        console.print(f"  Saved: {anchor_path.relative_to(ROOT)}")
    except ComfyUIError as e:
        console.print(f"[red]Anchor failed: {e}[/red]")
        raise SystemExit(1)

    uploaded_anchor = client.upload_image(str(anchor_path))

    failed = []
    for idx, (denoise, slide_prompt) in enumerate(slides):
        console.print(f"\n[bold]Slide {idx+1}/{len(slides)}[/bold] denoise={denoise:.2f}")
        console.print(f"  {slide_prompt[:100]}{'...' if len(slide_prompt) > 100 else ''}")

        for cand in range(candidates):
            slide_seed = random.randint(1, 2**31 - 1)
            base_path = inter / f"slide_{idx:02d}_cand_{cand}_base.png"
            final_path = out_dir / f"slide_{idx:02d}_cand_{cand}.png"

            try:
                # Stage 2: img2img off anchor — person+BG rendered together
                wf2 = _inject_flux_img2img(
                    load_workflow(str(ROOT / "workflows" / "flux_img2img.json")),
                    slide_prompt, uploaded_anchor, denoise, slide_seed,
                )
                _run_and_save(client, wf2, base_path)

                # Stage 3: ReActor face swap
                uploaded_target = client.upload_image(str(base_path))
                wf3 = _inject_faceswap(
                    load_workflow(str(ROOT / "workflows" / "faceswap_reactor.json")),
                    uploaded_face, uploaded_target,
                )
                _run_and_save(client, wf3, final_path, timeout=180)

                console.print(f"  [green]cand {cand}[/green] -> {final_path.name} (seed {slide_seed})")

            except (ComfyUIError, Exception) as e:
                console.print(f"  [red]cand {cand} FAILED: {e}[/red]")
                failed.append((idx, cand, str(e)))

    console.print(f"\n[bold green]Done.[/bold green] {out_dir.relative_to(ROOT)}")
    if failed:
        console.print(f"[yellow]Failed: {len(failed)}[/yellow]")
        for idx, cand, err in failed:
            console.print(f"  slide {idx} cand {cand}: {err}")


if __name__ == "__main__":
    main()
