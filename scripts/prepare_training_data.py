#!/usr/bin/env python3
"""Prepare character/seeds/ images for cloud LoRA training.

Steps:
  1. --validate   : check 8+8+8 split, PNG format, 1024×1024 size, diversity
  2. (default)    : resize/convert all images to 1024×1024 PNG
  3. --caption-style [sdxl|flux] : generate caption .txt files
  4. --zip-only   : package training_data/ into training_data.zip + kohya_config.toml

Usage:
    python scripts/prepare_training_data.py --validate
    python scripts/prepare_training_data.py --caption-style sdxl
    # → review/edit character/seeds/**/*.txt captions
    python scripts/prepare_training_data.py --zip-only
"""

import hashlib
import shutil
import sys
import textwrap
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

try:
    import click
    import yaml
    from PIL import Image, ImageOps
    from rich.console import Console
    from rich.table import Table
except ImportError:
    print("Missing dependencies. Run: pip install click PyYAML Pillow rich")
    sys.exit(1)

console = Console()

CATEGORIES = ["closeup", "medium", "fullbody"]
TARGET_SIZE = (1024, 1024)
MIN_PER_CATEGORY = 6
TARGET_PER_CATEGORY = 8
MIN_TOTAL = 24

# Tags stripped from SDXL captions (identity descriptors — isolation rule)
FACE_HAIR_TAGS = {
    "brown eyes", "black eyes", "blue eyes", "green eyes", "hazel eyes",
    "short hair", "long hair", "black hair", "brown hair", "blonde hair",
    "red hair", "straight hair", "wavy hair", "curly hair",
    "asian", "caucasian", "east asian", "south asian",
    "sharp eyes", "detailed face", "beautiful face", "pretty face",
    "young woman", "girl", "1girl", "woman", "female",
    "cute", "attractive", "gorgeous",
}

KOHYA_TOML = """[general]
enable_bucket = true
bucket_reso_steps = 64
min_bucket_reso = 256
max_bucket_reso = 1024

[datasets.[[datasets]].subsets]
image_dir = "./train_data"
caption_extension = ".txt"
num_repeats = 10
keep_tokens = 1

[model_arguments]
pretrained_model_name_or_path = "RunDiffusion/Juggernaut-X-v10"
v2 = false

[network_arguments]
network_module = "networks.lora"
network_dim = 32
network_alpha = 32

[optimizer_arguments]
optimizer_type = "Prodigy"
optimizer_args = ["d_coef=2", "use_bias_correction=True", "safeguard_warmup=True"]
learning_rate = 1e-4

[training_arguments]
output_dir = "./output"
output_name = "KaviB_v1_Prod"
save_model_as = "safetensors"
max_train_steps = 2400
noise_offset = 0.0357
min_snr_gamma = 5.0
mixed_precision = "fp16"
xformers = true
cache_latents = true
cache_latents_to_disk = true
gradient_checkpointing = true

[sample_prompts]
sample_every_n_steps = 200
sample_sampler = "dpmpp_2m"
sample_prompts = [
  "a photo of KaviB standing in a crowded city street, evening lighting",
  "a close up of KaviB laughing, wearing a yellow hat",
  "KaviB sitting on a park bench, full body shot"
]
"""


def image_histogram_signature(img: Image.Image) -> tuple:
    """Coarse histogram signature for diversity checking."""
    small = img.resize((32, 32)).convert("L")
    hist = small.histogram()
    # Bin into 8 buckets
    bucket_size = len(hist) // 8
    return tuple(sum(hist[i:i+bucket_size]) for i in range(0, len(hist), bucket_size))


def histogram_distance(a: tuple, b: tuple) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / max(sum(a), 1)


def check_diversity(images: list[Path], category: str) -> list[str]:
    """Warn if >4 images in a category share similar background/lighting."""
    warnings = []
    if len(images) < 2:
        return warnings
    sigs = []
    for img_path in images:
        try:
            img = Image.open(img_path)
            sigs.append((img_path, image_histogram_signature(img)))
        except Exception:
            pass

    similar_groups: list[list[Path]] = []
    used = set()
    for i, (path_a, sig_a) in enumerate(sigs):
        if i in used:
            continue
        group = [path_a]
        for j, (path_b, sig_b) in enumerate(sigs[i+1:], i+1):
            if j not in used and histogram_distance(sig_a, sig_b) < 0.15:
                group.append(path_b)
                used.add(j)
        if len(group) > 4:
            warnings.append(
                f"  [yellow]DIVERSITY WARNING[/yellow] {category}: "
                f"{len(group)} images appear to share similar background/lighting. "
                f"Vary your scenes for a more robust LoRA."
            )
        used.add(i)
    return warnings


def get_images(category_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(p for p in category_dir.iterdir()
                  if p.suffix.lower() in exts and not p.name.startswith("."))


def validate_dataset(seeds_dir: Path) -> bool:
    console.print("\n[bold]Dataset Validation[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Category")
    table.add_column("Count", justify="right")
    table.add_column("Status")
    table.add_column("Issues")

    all_ok = True
    total = 0
    all_warnings = []

    for cat in CATEGORIES:
        cat_dir = seeds_dir / cat
        images = get_images(cat_dir) if cat_dir.exists() else []
        count = len(images)
        total += count
        issues = []

        if count < MIN_PER_CATEGORY:
            issues.append(f"need at least {MIN_PER_CATEGORY}")
            all_ok = False
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    if img.size != TARGET_SIZE:
                        issues.append(f"{img_path.name}: {img.size[0]}×{img.size[1]} (need 1024×1024)")
                    if img.mode not in ("RGB", "RGBA"):
                        issues.append(f"{img_path.name}: mode={img.mode}")
                    if img.format and img.format.upper() not in ("PNG",):
                        issues.append(f"{img_path.name}: format={img.format} (prefer PNG)")
            except Exception as e:
                issues.append(f"{img_path.name}: cannot open ({e})")

        warnings = check_diversity(images, cat)
        all_warnings.extend(warnings)

        status = "[green]OK[/green]" if count >= MIN_PER_CATEGORY and not issues else "[red]FAIL[/red]"
        table.add_row(cat, str(count), status, "; ".join(issues[:2]) if issues else "")

    console.print(table)
    console.print(f"\nTotal images: {total} / {MIN_TOTAL} required")

    for w in all_warnings:
        console.print(w)

    if total < MIN_TOTAL:
        console.print(
            f"\n[red]FAIL:[/red] Need {MIN_TOTAL} images total ({TARGET_PER_CATEGORY} per category). "
            f"Run bootstrap_seeds.py to generate candidates, then curate the best."
        )
        all_ok = False
    elif all_ok:
        console.print("\n[green]PASS[/green] — Dataset ready for processing.")

    return all_ok


def process_images(seeds_dir: Path, out_dir: Path) -> list[Path]:
    """Resize and convert all seed images to 1024×1024 PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = []
    for cat in CATEGORIES:
        cat_dir = seeds_dir / cat
        out_cat_dir = out_dir / cat
        out_cat_dir.mkdir(parents=True, exist_ok=True)
        for img_path in get_images(cat_dir):
            try:
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB")
                    img_fit = ImageOps.fit(img_rgb, TARGET_SIZE, Image.LANCZOS)
                    out_path = out_cat_dir / (img_path.stem + ".png")
                    img_fit.save(out_path, format="PNG")
                    processed.append(out_path)
                    console.print(f"  [dim]processed[/dim] {cat}/{img_path.name} → {out_path.name}")
            except Exception as e:
                console.print(f"  [red]ERROR[/red] {img_path.name}: {e}")
    return processed


def generate_caption_sdxl(img_path: Path, category: str) -> str:
    """Generate a WD14-style caption skeleton, stripping face/hair identity tags."""
    cat_tags = {
        "closeup": "close-up portrait, upper body",
        "medium": "waist-up, medium shot",
        "fullbody": "full body, standing",
    }
    scene_stub = cat_tags.get(category, "portrait")
    # Base tag list — user should expand with actual clothing/setting details
    tags = [
        "KaviB",
        scene_stub,
        "photorealistic",
        "natural lighting",
        "looking at viewer",
        "detailed background",
    ]
    return ", ".join(tags)


def generate_caption_flux(img_path: Path, category: str) -> str:
    """Generate a JoyCaption-style natural language skeleton."""
    cat_desc = {
        "closeup": "a close-up photo showing the face and upper shoulders",
        "medium": "a waist-up photo",
        "fullbody": "a full-body photo",
    }
    desc = cat_desc.get(category, "a photo")
    return (
        f"A photo of KaviB, {desc}. "
        f"[DESCRIBE: clothing, setting, lighting, pose, mood — "
        f"NOT face, hair, or skin descriptors]"
    )


def write_captions(processed_images: list[Path], caption_style: str) -> None:
    console.print(f"\n[bold]Writing {caption_style.upper()} captions...[/bold]")
    console.print(
        "[yellow]IMPORTANT:[/yellow] Review and edit every .txt file before zipping. "
        "Remove any face/hair/identity descriptors. Only clothing, setting, pose, lighting."
    )
    for img_path in processed_images:
        cat = img_path.parent.name
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            console.print(f"  [dim]SKIP[/dim] {txt_path.name} (already exists)")
            continue
        if caption_style == "sdxl":
            caption = generate_caption_sdxl(img_path, cat)
        else:
            caption = generate_caption_flux(img_path, cat)
        txt_path.write_text(caption, encoding="utf-8")
        console.print(f"  [green]OK[/green] {txt_path.name}")


def create_zip_and_toml(out_dir: Path) -> None:
    flat_dir = PROJECT_ROOT / "training_data"
    flat_dir.mkdir(exist_ok=True)

    # Flatten all category subfolders into training_data/
    for cat in CATEGORIES:
        cat_dir = out_dir / cat
        if not cat_dir.exists():
            continue
        for f in cat_dir.iterdir():
            if f.suffix.lower() in {".png", ".txt"}:
                shutil.copy2(f, flat_dir / f.name)

    zip_path = PROJECT_ROOT / "training_data.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(flat_dir.iterdir()):
            zf.write(f, f"train_data/{f.name}")
    console.print(f"  [green]OK[/green] {zip_path} ({zip_path.stat().st_size // 1024}KB)")

    toml_path = PROJECT_ROOT / "kohya_config.toml"
    toml_path.write_text(KOHYA_TOML, encoding="utf-8")
    console.print(f"  [green]OK[/green] {toml_path}")

    console.print(
        f"\n[bold]Ready for cloud training.[/bold]\n"
        f"Upload to RunPod or Civitai:\n"
        f"  {zip_path}\n"
        f"  {toml_path}\n"
        f"See setup/train_lora_guide.md for step-by-step instructions."
    )


@click.command()
@click.option("--validate", is_flag=True, help="Validate dataset only (no processing)")
@click.option("--caption-style", default="sdxl", type=click.Choice(["sdxl", "flux"]),
              show_default=True, help="Caption generation style")
@click.option("--zip-only", is_flag=True, help="Skip processing; just zip training_data/ and write TOML")
@click.option("--config", "config_path", default="config.yaml", show_default=True)
def main(validate, caption_style, zip_only, config_path):
    with open(PROJECT_ROOT / config_path) as f:
        cfg = yaml.safe_load(f)

    seeds_dir = PROJECT_ROOT / cfg["character"]["seeds_dir"]
    out_dir = PROJECT_ROOT / "training_data_processed"

    if validate:
        ok = validate_dataset(seeds_dir)
        sys.exit(0 if ok else 1)

    if zip_only:
        console.print("\n[bold]Packaging training data...[/bold]")
        create_zip_and_toml(out_dir)
        return

    console.print("\n[bold]Processing seed images...[/bold]")
    ok = validate_dataset(seeds_dir)
    if not ok:
        console.print("\n[red]Fix dataset issues before processing.[/red]")
        sys.exit(1)

    processed = process_images(seeds_dir, out_dir)
    console.print(f"\n  Processed {len(processed)} images → {out_dir}")

    write_captions(processed, caption_style)

    console.print(
        f"\n[bold]Next steps:[/bold]\n"
        f"  1. Review and edit every .txt file in {out_dir}/**/*.txt\n"
        f"     Strip any face/hair/identity descriptors.\n"
        f"  2. Run: python scripts/prepare_training_data.py --zip-only\n"
        f"     This creates training_data.zip and kohya_config.toml."
    )


if __name__ == "__main__":
    main()
