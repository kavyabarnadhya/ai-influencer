"""
Bootstrap diverse v1 LoRA seed candidates for Ananya v2 dataset assembly.

Generates 30-40 candidate images covering all diversity gates from the v2 seed matrix:
outfit × hair × scene × focal × DOF × aesthetic × pose. Curate down to 5-8 keepers.

Usage:
    python scripts/bootstrap_seeds_v2.py
    python scripts/bootstrap_seeds_v2.py --count 40 --seed 1000
    python scripts/bootstrap_seeds_v2.py --dry-run
"""

import json
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import click
import cv2
import numpy as np
import yaml
from PIL import Image
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))
from comfyui_api import ComfyUIClient, ComfyUIError, find_comfyui_port, inject_workflow_values, load_workflow

console = Console()
ROOT = Path(__file__).parent.parent

FACE_REF = ROOT / "character" / "ananya" / "seeds_v2" / "face_ref_v2.png"

# ---------------------------------------------------------------------------
# Prompt component pools — drawn from v2_seed_matrix.md diversity gates
# ---------------------------------------------------------------------------

SHOT_TYPES = [
    # (label, shot_phrase, focal_phrase, weight)
    ("closeup_85mm",   "close-up portrait, bust shot",                                 "compressed telephoto portrait, 85mm equivalent, background compression",   3),
    ("closeup_50mm",   "close-up portrait, chin to chest",                             "50mm natural portrait framing, minimal compression",                       2),
    ("medium_35mm",    "waist-up shot",                                                 "35mm lifestyle framing, natural perspective, subject prominent in scene",   3),
    ("medium_50mm",    "waist-up shot, three-quarter view",                             "50mm natural portrait framing, minimal compression",                       2),
    ("fullbody_35mm",  "full body shot",                                                "35mm lifestyle framing, natural perspective",                               2),
    ("fullbody_24mm",  "full body shot, wide environmental framing",                    "wide environmental framing, 24mm equivalent, slight perspective distortion",1),
    ("extreme_closeup","extreme close-up of face, forehead to chin",                   "compressed telephoto portrait, 85mm equivalent",                           1),
]

OUTFITS = [
    # (label, prompt)
    ("silk_saree",        "silk saree with draped pallu over left shoulder, gold zari border, fitted blouse, no midriff, traditional drape"),
    ("kurti_palazzo",     "knee-length cotton kurti, V-neck, three-quarter sleeves, palazzo pants, dupatta around neck"),
    ("lehenga",           "flared lehenga with mirror work, fitted choli, embroidered dupatta over shoulder, ethnic festive"),
    ("jeans_top",         "high-waisted dark blue jeans, fitted white cotton crew-neck top, tucked in, casual"),
    ("bodycon_midi",      "emerald green fitted bodycon midi dress, scoop neck, sleeveless, one-piece, no midriff"),
    ("blazer_cami",       "tailored black blazer over champagne satin cami, smart casual"),
    ("sundress",          "floral sundress, midi length, thin straps, summer casual"),
    ("coord_set",         "off-white linen co-ord set, wide-leg trousers, cropped top with wide neckline, modest"),
    ("white_tee",         "plain white fitted crew-neck t-shirt, dark jeans, casual everyday"),
    ("plain_wrap",        "dusty rose wrap midi skirt, simple white tucked blouse, soft feminine"),
    ("salwar_kameez",     "long anarkali kurta, churidar pants, sheer dupatta around neck, semi-formal Indian"),
]

HAIR_STYLES = [
    # (label, prompt)
    ("loose_waves_side",  "loose waves, side-parted, flowing past shoulders"),
    ("loose_waves_center","loose waves, center-parted, natural texture"),
    ("half_up",           "half-up half-down, top section pinned loosely, soft face framing strands"),
    ("low_bun",           "low bun, soft tendrils framing face, neat but relaxed"),
    ("high_bun",          "sleek high bun, no loose strands, clean"),
    ("ponytail",          "mid-height ponytail, slight volume at crown, soft"),
    ("braid",             "single loose braid over shoulder, natural texture"),
    ("wind_flip",         "loose waves mid-flip from wind, hair caught in motion, dynamic"),
    ("wet_texture",       "loose waves, slightly damp texture, natural and effortless"),
    ("messy_strands",     "loose waves with flyaways and messy strands, lived-in texture"),
    ("backlit_strands",   "loose waves, rim light catching individual hair strands, glowing edges"),
]

SCENES = [
    # (label, scene_phrase, aesthetic)
    ("hotel_lobby",    "luxury hotel lobby, Italian marble floors, warm tungsten chandelier light, high ceiling",       "soft luxury lifestyle"),
    ("rooftop_golden", "open rooftop terrace, distant Bengaluru city skyline, hazy golden hour pink sky",               "high-contrast rooftop portrait, dramatic shadows, bold golden light"),
    ("cafe_interior",  "modern concrete café interior, exposed brick, tropical greenery visible through window, warm tungsten pendant lights", "candid lifestyle photography, natural unstaged moment"),
    ("street_india",   "narrow leafy street, terracotta-tiled rooftops, warm street lamp glow, Bengaluru evening",     "candid phone camera realism, authentic"),
    ("studio_neutral", "softbox studio, neutral grey seamless paper backdrop, even diffused light",                    "editorial fashion photography, strong composition"),
    ("indoor_warm",    "modern mid-range Bengaluru apartment, white walls, warm LED panel light, urban domestic",       "indoor warm lifestyle, tungsten light, cozy ambient"),
    ("garden",         "Lalbagh botanical garden, lush tropical greenery, dappled sunlight through canopy",             "muted cloudy daylight, flat soft light, desaturated palette, indie editorial"),
    ("hotel_room",     "hotel room vanity mirror, warm bulb ring lights, white marble countertop",                     "soft luxury lifestyle, warm tones"),
    ("heritage_street","old-city narrow lane, weathered pastel walls, cobblestones, soft diffused cloudy daylight, North India heritage aesthetic", "UGC style, imperfect framing, natural light, authentic everyday India"),
]

DOF_OPTIONS = [
    # (label, prompt, weight)  — target 50% deep, 30% medium, 20% bokeh
    ("deep_focus",   "sharp background, environmental detail visible, f/8 deep focus, iPhone 15 Pro Max realism, candid phone camera shot", 5),
    ("medium_dof",   "moderate background blur, some background detail visible, f/2.8 portrait, DSLR natural",                             3),
    ("shallow_bokeh","shallow depth of field, strong background bokeh, f/1.4 editorial, subject sharp, background melted",                  2),
]

CAMERA_ANGLES = [
    "facing camera directly, eye level",
    "three-quarter view from camera left, eye level",
    "three-quarter view from camera right, eye level",
    "profile from camera left, eye level",
    "slight high angle, looking slightly down at subject",
    "slight low angle, looking slightly up at subject",
    "facing camera, slight dutch tilt, candid",
]

EMOTIONS = [
    "soft smile, warm expression",
    "neutral expression, direct gaze, serious",
    "candid laugh, eyes crinkled",
    "distracted, looking away from camera",
    "tired, flat expression, minimal makeup energy",
    "thoughtful, slight head tilt, soft gaze",
    "confident direct eye contact, no smile",
]

SENSOR_MODES = [
    "DSLR fashion photography, Sony A7IV quality, clean sensor, professional grade",
    "iPhone 15 Pro Max shot, candid, phone camera realism, natural skin tone rendering",
    "iPhone 15 Pro Max shot, candid, phone camera realism, natural skin tone rendering",
    "subtle film grain, analog texture, slightly desaturated, editorial film aesthetic",
    "handheld low-light mobile shot, visible grain, slight motion softness, candid",
]

JEWELRY_OPTIONS = [
    "tiny understated gold stud earrings, barely visible thin gold chain necklace, no rings",
    "small matte gold hoop earrings, delicate thin gold pendant necklace, no rings",
    "small pearl stud earrings, no necklace, no rings",
    "tiny gold huggie earrings, barely visible thin gold necklace, no rings",
    "no earrings, no necklace, no rings",
    "small subtle gold drop earrings, delicate thin gold chain, no rings",
]

NEGATIVE_BASE = (
    "ugly, deformed, bad anatomy, bad proportions, extra limbs, malformed hands, extra fingers, "
    "fused fingers, missing fingers, bad hands, watermark, text, signature, logo, "
    "plastic skin, waxy skin, over-smooth skin, airbrushed, artificial skin texture, "
    "heavy makeup, bridal, bindi, tikka, heavy traditional jewelry, chunky jewelry, "
    "revealing clothing, swimwear, lingerie, crop top showing midriff, "
    "cartoon, illustration, painting, 3d render, cgi, low quality, blurry"
)


def crop_face_for_ipadapter(image_path: Path, padding: float = 0.35) -> Path:
    """Crop to face bbox + padding. Returns path to temp PNG (square).

    Uses Haar cascade — fast, no extra deps. Falls back to center-crop if no
    face detected. Temp file lives for the process lifetime.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    h, w = img.shape[:2]

    if len(faces) == 0:
        # Fallback: top-center crop (portrait photos — face usually top-center)
        console.print("  [yellow]No face detected — using top-center crop fallback[/yellow]")
        crop_size = min(w, h // 2)
        x1 = (w - crop_size) // 2
        y1 = 0
        x2 = x1 + crop_size
        y2 = crop_size
    else:
        # Pick largest face
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        pad_x = int(fw * padding)
        pad_y = int(fh * padding)
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y)
        x2 = min(w, fx + fw + pad_x)
        y2 = min(h, fy + fh + pad_y)
        # Expand to square
        side = max(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)

    crop = img[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb).resize((512, 512), Image.LANCZOS)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    pil.save(tmp.name)
    return Path(tmp.name)


def weighted_choice(options):
    items = [item for item in options]
    weights = [item[-1] for item in items]  # weight is always last element
    chosen = random.choices(items, weights=weights, k=1)[0]
    return chosen


def build_prompt(shot_label: str, shot_phrase: str, focal_phrase: str,
                 outfit: str, hair: str, scene: str, aesthetic: str,
                 dof: str, angle: str, emotion: str, sensor: str,
                 jewelry: str, trigger: str = "") -> tuple[str, str]:
    positive = (
        f"{shot_phrase}, {focal_phrase}, "
        f"{angle}, {emotion}, "
        f"wearing {outfit}, "
        f"hair: {hair}, "
        f"{jewelry}, "
        f"{scene}, "
        f"{dof}, "
        f"{aesthetic}, "
        f"{sensor}, "
        f"sharp detailed face, photorealistic, masterpiece"
    )
    return positive, NEGATIVE_BASE


def generate_candidate_configs(count: int, base_seed: int) -> list[dict]:
    configs = []

    # Ensure at minimum 1 image per outfit (11 outfits)
    outfit_coverage = {o[0]: 0 for o in OUTFITS}
    hair_coverage = {h[0]: 0 for h in HAIR_STYLES}

    for i in range(count):
        seed = base_seed + i

        shot = weighted_choice(SHOT_TYPES)
        outfit = random.choice(OUTFITS)
        hair = random.choice(HAIR_STYLES)
        scene = random.choice(SCENES)
        dof = weighted_choice(DOF_OPTIONS)
        angle = random.choice(CAMERA_ANGLES)
        emotion = random.choice(EMOTIONS)
        sensor = random.choice(SENSOR_MODES)
        jewelry = random.choice(JEWELRY_OPTIONS)

        # Force outfit coverage: pick uncovered outfit if any remain
        uncovered_outfits = [o for o in OUTFITS if outfit_coverage[o[0]] == 0]
        if uncovered_outfits and i < len(OUTFITS):
            outfit = uncovered_outfits[0]

        outfit_coverage[outfit[0]] += 1
        hair_coverage[hair[0]] += 1

        positive, negative = build_prompt(
            shot_label=shot[0], shot_phrase=shot[1], focal_phrase=shot[2],
            outfit=outfit[1], hair=hair[1],
            scene=scene[1], aesthetic=scene[2],
            dof=dof[1], angle=angle, emotion=emotion,
            sensor=sensor, jewelry=jewelry,
            trigger="",
        )

        # Resolution by shot type
        if "extreme_closeup" in shot[0] or "closeup" in shot[0]:
            width, height = 1024, 1024
        elif "fullbody_24mm" in shot[0]:
            width, height = 832, 1216
        else:
            # Mix 1024² and 832×1216
            width, height = random.choice([(1024, 1024), (832, 1216), (1024, 1024)])

        configs.append({
            "index": i + 1,
            "seed": seed,
            "shot_label": shot[0],
            "outfit_label": outfit[0],
            "hair_label": hair[0],
            "scene_label": scene[0],
            "dof_label": dof[0],
            "width": width,
            "height": height,
            "positive": positive,
            "negative": negative,
        })

    return configs


@click.command()
@click.option("--count", default=35, show_default=True, help="Number of candidates to generate")
@click.option("--seed", "base_seed", default=None, type=int, help="Base seed (random if omitted)")
@click.option("--dry-run", is_flag=True, help="Print prompts, do not call ComfyUI")
@click.option("--workflow", default="bootstrap_ipadapter", show_default=True, help="Workflow name (no .json)")
@click.option("--steps", default=30, show_default=True)
@click.option("--cfg", default=7.5, show_default=True, type=float)
@click.option("--out-dir", default=None, help="Override output directory")
def main(count: int, base_seed: int | None, dry_run: bool, workflow: str,
         steps: int, cfg: float, out_dir: str | None):
    """Generate diverse IPAdapter-steered candidates for Ananya v2 (Juggernaut + face_ref_v2.png anchor)."""

    if base_seed is None:
        base_seed = random.randint(1, 2**31)

    console.print(f"[bold]Ananya v2 Bootstrap Seeds[/bold]")
    console.print(f"Count: {count} | Base seed: {base_seed} | Workflow: {workflow}")
    console.print(f"Face ref: [cyan]{FACE_REF.name}[/cyan]")

    configs = generate_candidate_configs(count, base_seed)

    if dry_run:
        table = Table(title="Candidate prompts (dry run)", show_lines=True)
        table.add_column("#", width=3)
        table.add_column("Shot", width=16)
        table.add_column("Outfit", width=16)
        table.add_column("Hair", width=16)
        table.add_column("Scene", width=16)
        table.add_column("DOF", width=12)
        table.add_column("Res", width=10)
        for c in configs:
            table.add_row(
                str(c["index"]),
                c["shot_label"],
                c["outfit_label"],
                c["hair_label"],
                c["scene_label"],
                c["dof_label"],
                f"{c['width']}×{c['height']}",
            )
        console.print(table)
        console.print("\n[dim]First candidate prompt:[/dim]")
        console.print(configs[0]["positive"])
        return

    # --- Find ComfyUI ---
    port = find_comfyui_port()
    if not port:
        console.print("[red]ComfyUI not found on any port. Start ComfyUI Desktop first.[/red]")
        raise SystemExit(1)
    console.print(f"ComfyUI found on port {port}")

    client = ComfyUIClient(host="127.0.0.1", port=port)

    # Load workflow
    workflow_path = ROOT / "workflows" / f"{workflow}.json"
    if not workflow_path.exists():
        console.print(f"[red]Workflow not found: {workflow_path}[/red]")
        raise SystemExit(1)

    # Output directory
    date_str = datetime.now().strftime("%Y-%m-%d")
    if out_dir:
        out_path = Path(out_dir)
    else:
        out_path = ROOT / "character" / "ananya" / "seeds_v2" / "experimental" / f"bootstrap_{date_str}_seed{base_seed}"
    out_path.mkdir(parents=True, exist_ok=True)

    # Save candidate manifest
    manifest_path = out_path / "candidates_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"base_seed": base_seed, "count": count, "workflow": workflow, "candidates": configs}, f, indent=2)
    console.print(f"Manifest: {manifest_path}")

    # Crop face ref to face-only bbox before upload — prevents hair style bleed
    if not FACE_REF.exists():
        console.print(f"[red]Face reference not found: {FACE_REF}[/red]")
        raise SystemExit(1)
    console.print(f"Cropping face from: {FACE_REF.name}...")
    face_crop_path = crop_face_for_ipadapter(FACE_REF)
    console.print(f"Uploading face crop (512x512)...")
    uploaded_face_ref = client.upload_image(str(face_crop_path))

    # --- Generate ---
    failed = []
    for c in configs:
        idx = c["index"]
        console.print(f"\n[dim]Candidate {idx}/{count} — {c['shot_label']} / {c['outfit_label']} / {c['hair_label']}[/dim]")

        wf = load_workflow(str(workflow_path))
        # Performance Optimization: Skip cache propagation on the final injection
        # to avoid an extra dictionary copy in client.submit_workflow().
        wf = inject_workflow_values(wf, {
            "_claude_inject_prompt": {"inputs.text": c["positive"]},
            "_claude_inject_negative": {"inputs.text": c["negative"]},
            "_claude_inject_seed": {"inputs.seed": c["seed"], "inputs.steps": steps, "inputs.cfg": cfg},
            "_claude_inject_ipadapter_image": {"inputs.image": uploaded_face_ref},
            "_claude_inject_ipadapter_strength": {"inputs.weight": 0.6},
            "_claude_inject_latent": {"inputs.width": c["width"], "inputs.height": c["height"]},
        }, propagate_cache=False)

        try:
            prompt_id = client.submit_workflow(wf)
            image_refs = client.wait_for_completion(prompt_id, timeout=120)
            if not image_refs:
                console.print(f"  [yellow]No images returned for candidate {idx}[/yellow]")
                failed.append(idx)
                continue

            filename = f"cand_{idx:03d}_{c['shot_label']}_{c['outfit_label']}_{c['seed']}.png"
            save_path = out_path / filename
            img_bytes = client.download_image(
                image_refs[0]["filename"],
                image_refs[0].get("subfolder", ""),
                image_refs[0].get("type", "output"),
            )
            with open(save_path, "wb") as f:
                f.write(img_bytes)
            console.print(f"  Saved: {save_path.name}")

        except ComfyUIError as e:
            console.print(f"  [red]Error: {e}[/red]")
            failed.append(idx)

    # Summary
    console.print(f"\n[bold green]Done.[/bold green] {count - len(failed)}/{count} generated.")
    if failed:
        console.print(f"[yellow]Failed candidates: {failed}[/yellow]")
    console.print(f"\nOutput: {out_path}")
    console.print("\n[bold]Next: open output dir, curate 5-8 keepers -> copy to seeds_v2/training_canonical/[/bold]")
    console.print("Curation criteria: sharp face, good skin texture, correct outfit, no artifacts, diverse coverage")


if __name__ == "__main__":
    main()
