import random
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

sys.path.insert(0, str(Path(__file__).parent))
from comfyui_api import ComfyUIClient, ComfyUIError, inject_workflow_values, load_workflow

console = Console()
ROOT = Path(__file__).parent.parent

MODE_PREFIXES = {
    "closeup": "portrait, close-up face shot, head and shoulders only",
    "medium": "waist-up shot, medium portrait",
    "fullbody": "full body shot, full length portrait",
}

VARIETY_POOL = {
    "closeup": [
        "soft 3/4 angle, warm golden hour window light, blurred bokeh background",
        "direct gaze, moody bedside lamp, hotel room, intimate warm tones",
        "slight side turn, soft natural window light, bright airy interior",
        "candid over-shoulder look, outdoor golden hour, rooftop",
        "downward gaze, soft overhead diffused light, sunlit cafe interior",
        "frontal, split dramatic side lighting, dark moody background",
        "slight chin tilt, champagne warm ambience, hotel room intimate",
        "3/4 turn, misty morning outdoor light, balcony railing",
        "direct gaze, cinematic corridor side light, dark gradient background",
        "candid half-smile, dappled garden light, outdoor natural",
        "looking away, soft warm lamp, mirror reflection visible in background",
        "neutral confidence, overcast soft outdoor light, city street",
        "eyes closed, golden rim backlight, outdoor sunset silhouette edge",
        "slight smirk, warm bathroom vanity light, intimate mirror selfie feel",
        "frontal neutral, cool blue-hour window light, modern interior",
        "candid laugh, warm string light bokeh background, rooftop evening",
        # Gap entries: moody/editorial, different hair structure, different neckline, outdoor contrast
        "hair pulled into loose bun, dark editorial background, dramatic split side lighting, bold structured neckline",
        "hair swept to one side, strong outdoor daylight, high contrast, natural imperfect skin texture",
        "dark moody background, deep side lighting, editorial fashion, strong shadow on one side of face",
        "messy undone hair, overcast outdoor light, candid mid-movement feel, unpolished natural",
        "high neck top, cool blue-grey tones, minimalist modern interior",
        "off-shoulder, warm terracotta wall background, golden mediterranean afternoon light",
        "textured updo, deep green garden background, dappled natural leaf light",
        "dark charcoal background, hard rim light only, cinematic dramatic portrait",
    ],
    "medium": [
        # Lifestyle / travel
        "hotel lobby, standing confident, tailored blazer over fitted top, cool architectural interior light, editorial lifestyle photography",
        "sunlit cafe, seated with iced coffee, white linen kurta, dappled window light, candid lifestyle portrait",
        "rooftop golden hour, terracotta co-ord set, wind in hair, looking slightly off-camera, warm backlight, editorial",
        "Delhi street style, oversized jacket over fitted top, leaning casually on wall, warm afternoon urban light",
        "balcony morning, oversized knit sweater, relaxed pose, soft overcast daylight, natural candid portrait",
        "garden patio brunch, floral midi dress, seated at table, soft dappled sunlight, warm lifestyle photography",
        "airport departure hall, beige trench over casual chic outfit, tote bag, confident standing pose, cool overhead light",
        "misty mountain viewpoint, oversized sweater and jeans, seated on railing, foggy green backdrop, candid travel portrait",
        # Premium / intimate
        "hotel room window light, satin slip dress, seated by curtains, soft morning light, elegant intimate editorial",
        "bathroom vanity, wet-hair glam, fitted sleeveless blouse, warm mirror light, tasteful premium portrait",
        "rooftop evening, deep plunge dress, leaning on railing, city bokeh, moody glam editorial",
        "private glam shoot, silk wrap top, direct gaze, warm ambient light, subscriber-style premium portrait",
        # Indian fashion / ethnic
        "festive veranda, silk kurta set, warm string lights behind her, soft evening glow, elegant Indian lifestyle",
        "haveli courtyard, block-print co-ord set, seated on steps, natural golden light, refined ethnic editorial",
        "casual saree look, modern sleeveless blouse, standing by balcony, warm late afternoon light, contemporary Indian fashion portrait",
        "wedding guest styling, embroidered minimal lehenga, outdoor venue background, candid laugh, editorial lifestyle",
    ],
    "fullbody": [
        # Lifestyle / travel
        "airport departure hall, walking with tote bag, beige trench coat and jeans, cool overhead light, full body lifestyle shot",
        "Delhi street, casual walking, oversized blazer and flared pants, warm urban afternoon light, candid editorial",
        "rooftop golden hour, silk co-ord set, standing at railing, warm directional backlight, full body fashion shot",
        "beach shoreline, white linen dress, walking barefoot, warm sand and ocean backdrop, relaxed travel lifestyle",
        "Himalayan viewpoint, oversized knit sweater and boots, standing against misty green mountain backdrop, candid full body shot",
        "outdoor market lane, cropped embroidered jacket and trousers, browsing stalls, warm afternoon light, urban Indian fashion",
        "hotel lobby, trench over kurta set, striding confidently, cool interior architecture, editorial travel look",
        "indoor atelier, silk co-ord, standing naturally near worktable, soft diffused overhead light, fashion lifestyle shot",
        # Premium / glam
        "hotel corridor, backless fitted dress, walking away then turning slightly, cinematic side light, premium glam full body",
        "rooftop night, deep red bodycon dress, standing centered, city bokeh behind, moody editorial glam",
        "balcony dusk, fitted satin dress, leaning on railing, cool evening air, luxury subscriber-style fashion shot",
        "hotel room window, champagne slip dress, standing near curtain light, soft warm side light, intimate editorial",
        # Indian fashion / ethnic
        "contemporary saree look, modern blouse, full body standing pose, balcony golden hour, elegant Indian fashion",
        "festive courtyard, embroidered anarkali, natural standing pose, warm string lights, refined celebration styling",
        "Rajasthan haveli archway, block-print saree, desert golden light, full body editorial composition",
        "wedding guest outfit, minimal lehenga with modern styling, walking through venue, candid full body luxury fashion shot",
    ],
}


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


@click.command()
@click.option("--mode", type=click.Choice(["closeup", "medium", "fullbody"]), required=True, help="Shot type")
@click.option("--count", default=16, show_default=True, help="Number of candidates to generate")
@click.option("--character", default="ananya", show_default=True, help="Character to generate [ananya|kavib]")
@click.option("--output-dir", default=None, type=click.Path(path_type=Path), help="Override output directory (default: character seeds dir)")
@click.option("--ipadapter-ref", default=None, type=click.Path(exists=True, path_type=Path), help="Reference image for IP-Adapter steering (uses bootstrap_ipadapter workflow)")
@click.option("--ipadapter-strength", default=None, type=float, help="Override IP-Adapter weight (default: from config). Lower = more prompt variety, less aesthetic lock.")
@click.option("--dry-run", is_flag=True, default=False, help="Print prompts only, do not generate")
def main(mode: str, count: int, character: str, output_dir: Path | None, ipadapter_ref: Path | None, ipadapter_strength: float | None, dry_run: bool):
    cfg = load_config()
    char_cfg = load_character(cfg, character)
    comfy_cfg = cfg["comfyui"]
    gen_cfg = cfg["generation"]

    client = ComfyUIClient(comfy_cfg["host"], comfy_cfg["port"])
    if not dry_run and not client.is_running():
        console.print("[red]ComfyUI is not running. Start it first.[/red]")
        raise SystemExit(1)

    # Select workflow: IP-Adapter steered or plain bootstrap
    if dry_run:
        workflow_data = None
        uploaded_ref = None
    elif ipadapter_ref is not None:
        workflow_name = "bootstrap_ipadapter"
        console.print(f"[cyan]IP-Adapter mode: uploading reference image {ipadapter_ref.name}...[/cyan]")
        try:
            uploaded_ref = client.upload_image(str(ipadapter_ref))
        except Exception as e:
            console.print(f"[red]Failed to upload reference image: {e}[/red]")
            raise SystemExit(1)
        console.print(f"[dim]Reference uploaded as: {uploaded_ref}[/dim]")
    elif not dry_run:
        workflow_name = "bootstrap_seeds"
        uploaded_ref = None

    if not dry_run:
        workflow_path = ROOT / cfg["paths"]["workflows_dir"] / f"{workflow_name}.json"
        if not workflow_path.exists():
            console.print(f"[red]Workflow not found: {workflow_path}[/red]")
            raise SystemExit(1)
        workflow_data = load_workflow(str(workflow_path))

    out_dir = output_dir if output_dir is not None else ROOT / char_cfg["seeds_dir"] / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_prefix = MODE_PREFIXES[mode]
    base_prompt = (ROOT / char_cfg["base_prompt_file"]).read_text(encoding="utf-8").strip()

    pool = VARIETY_POOL[mode][:]
    random.shuffle(pool)

    if ipadapter_ref:
        active_strength = ipadapter_strength if ipadapter_strength is not None else char_cfg.get("ipadapter_strength", 0.6)
        label = f"IP-Adapter @ {active_strength}"
    elif dry_run:
        label = "dry-run"
    else:
        label = "plain"
    console.print(f"[cyan]Generating {count} {mode} candidates for {character} ({label}) -> {out_dir}[/cyan]")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task(f"Bootstrap {mode}", total=count)

        # Queuing Optimization: Submit all variations first to hide latency.
        pending_prompts: list[tuple[str, int, int]] = []

        for i in range(count):
            context = pool[i % len(pool)]
            # Bootstrap: full physical description + mode framing + per-image context. No trigger word — pre-LoRA.
            full_prompt = f"{base_prompt}, {mode_prefix}, {context}"
            seed = random.randint(0, 2**32 - 1)

            if dry_run:
                console.print(f"[dim][{i+1:02d}] {context}[/dim]")
                progress.advance(task)
                continue

            overrides = {
                "_claude_inject_prompt": {"inputs.text": full_prompt},
                "_claude_inject_negative": {"inputs.text": gen_cfg["negative_prompt"]},
                "_claude_inject_seed": {"inputs.seed": seed, "inputs.steps": gen_cfg["steps"], "inputs.cfg": gen_cfg["cfg"]},
                "_claude_inject_latent": {"inputs.width": gen_cfg["width"], "inputs.height": gen_cfg["height"]},
                "_claude_inject_checkpoint": {"inputs.ckpt_name": cfg["models"]["checkpoint"]},
            }

            if uploaded_ref is not None:
                strength = ipadapter_strength if ipadapter_strength is not None else char_cfg.get("ipadapter_strength", 0.6)
                overrides["_claude_inject_ipadapter_image"] = {"inputs.image": uploaded_ref}
                overrides["_claude_inject_ipadapter_strength"] = {"inputs.weight": strength}

            patched = inject_workflow_values(workflow_data, overrides)

            try:
                prompt_id = client.submit_workflow(patched)
                pending_prompts.append((prompt_id, seed, i))
            except ComfyUIError as e:
                console.print(f"[yellow]Submission {i+1} failed: {e}[/yellow]")
                progress.advance(task)

        # Download pass
        for prompt_id, seed, idx in pending_prompts:
            try:
                images = client.wait_for_completion(prompt_id, timeout=comfy_cfg["timeout"])
                for img_meta in images:
                    img_bytes = client.download_image(
                        img_meta["filename"], img_meta.get("subfolder", ""), img_meta.get("type", "output")
                    )
                    suffix = "_ipadapter" if ipadapter_ref else ""
                    filename = f"seed_{mode}{suffix}_{idx+1:03d}_{seed}.png"
                    dest = out_dir / filename
                    dest.write_bytes(img_bytes)
            except ComfyUIError as e:
                console.print(f"[yellow]Completion {idx+1} failed: {e}[/yellow]")

            progress.advance(task)

    if dry_run:
        console.print(f"[green]Dry run complete — {count} prompts printed above.[/green]")
        return

    saved = list(out_dir.glob("*.png"))
    console.print(f"[green]{len(saved)} candidates saved to {out_dir}[/green]")
    if output_dir is not None:
        console.print("[yellow]Next: review images, pick best 20-30 as your reference board.[/yellow]")
        console.print(f"[yellow]Then re-run with --ipadapter-ref <best_image> to generate steered seeds.[/yellow]")
    else:
        console.print("[yellow]Next: manually pick the best 8 images and keep only those.[/yellow]")
        console.print(f"[yellow]Then run: python scripts/prepare_training_data.py --character {character} --validate[/yellow]")


if __name__ == "__main__":
    main()
