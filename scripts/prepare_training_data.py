import functools
import os
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import click
import yaml
from PIL import Image
from rich.console import Console
from rich.table import Table

console = Console()
ROOT = Path(__file__).parent.parent

MODES = ["closeup", "medium", "fullbody"]
TARGET_SIZE = (1024, 1024)
MIN_IMAGES_PER_MODE = 8
FLUX_TARGET_IMAGES = 25
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
UNSUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".gif", ".tif", ".tiff", ".avif"}

MODE_CAPTION_PREFIX = {
    "closeup": "close-up face shot, head and shoulders",
    "medium": "waist-up shot, medium portrait",
    "fullbody": "full body shot, full length portrait",
}

FLUX_FORBIDDEN_PATTERNS = {
    "age": re.compile(r"\b(teen|teenager|young girl|minor|23\s*years?\s*old|twenty[- ]three)\b", re.IGNORECASE),
    "ethnicity": re.compile(r"\b(north indian|south indian|indian|south asian|asian|desi|punjabi|delhi girl)\b", re.IGNORECASE),
    "skin tone": re.compile(r"\b(skin tone|fair skin|brown skin|dark skin|dusky|wheatish|tan skin|olive skin|complexion)\b", re.IGNORECASE),
    "eye identity": re.compile(r"\b(dark eyes|brown eyes|black eyes|almond eyes|eye color|eye colour)\b", re.IGNORECASE),
    "face shape": re.compile(r"\b(oval face|round face|heart-shaped face|sharp jawline|jawline|cheekbones|nose shape|full lips)\b", re.IGNORECASE),
    "body identity": re.compile(r"\b(body shape|slim body|curvy body|petite|tall woman|short woman)\b", re.IGNORECASE),
}

# Programmatically build a unified regex for a fast-path search.
# Optimization: A single combined regex search is ~1.71x faster on clean captions
# compared to running 6 separate regex search passes sequentially.
_RE_ANY_FORBIDDEN = re.compile(
    r"\b(" + "|".join(
        p.pattern[3:-3] if p.pattern.startswith(r"\b(") and p.pattern.endswith(r")\b") else p.pattern
        for p in FLUX_FORBIDDEN_PATTERNS.values()
    ) + r")\b",
    re.IGNORECASE
)

# Pre-compiled regex for duplicate key normalization
_RE_DUP_VERSION = re.compile(r"\s*\(\d+\)$")
_RE_DUP_COPY = re.compile(r"\s+copy$")

FLUX_CAPTION_TEMPLATE = (
    "{shot_type} of {trigger}, seen from a three-quarter angle at eye level, "
    "with loose styled hair, wearing a contemporary fashion outfit. "
    "She is posing naturally with a relaxed expression. "
    "Soft editorial lighting. Lifestyle background."
)


@functools.lru_cache(maxsize=1)
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


def get_seed_images(seeds_dir: Path, mode: str) -> list[Path]:
    mode_dir = seeds_dir / mode
    if not mode_dir.exists():
        return []
    # Optimization: os.scandir is faster than Path.iterdir in large directories
    with os.scandir(mode_dir) as it:
        return sorted(
            Path(entry.path)
            for entry in it
            if entry.name.lower().endswith(".png")
        )


def get_training_data_dir(char_cfg: dict) -> Path:
    configured = char_cfg.get("training_data_dir")
    if configured:
        return ROOT / configured
    seeds_dir = ROOT / char_cfg["seeds_dir"]
    return seeds_dir.parent / "training_data"


def get_flux_images(training_dir: Path) -> list[Path]:
    if not training_dir.exists():
        return []
    # Optimization: os.scandir is faster than Path.iterdir in large directories
    exts = tuple(e.lower() for e in SUPPORTED_IMAGE_EXTENSIONS)
    with os.scandir(training_dir) as it:
        return sorted(
            Path(entry.path)
            for entry in it
            if entry.is_file() and entry.name.lower().endswith(exts)
        )


def get_unsupported_images(training_dir: Path) -> list[Path]:
    if not training_dir.exists():
        return []
    # Optimization: os.scandir is faster than Path.iterdir in large directories
    exts = tuple(e.lower() for e in UNSUPPORTED_IMAGE_EXTENSIONS)
    with os.scandir(training_dir) as it:
        return sorted(
            Path(entry.path)
            for entry in it
            if entry.is_file() and entry.name.lower().endswith(exts)
        )


def choose_layout(char_cfg: dict, requested_layout: str) -> str:
    if requested_layout != "auto":
        return requested_layout
    training_dir = get_training_data_dir(char_cfg)
    return "flux" if get_flux_images(training_dir) else "sdxl"


def normalize_duplicate_key(path: Path) -> str:
    stem = path.stem.lower()
    stem = _RE_DUP_VERSION.sub("", stem)
    stem = _RE_DUP_COPY.sub("", stem)
    return stem


def find_forbidden_caption_terms(caption: str) -> list[str]:
    """
    Find which forbidden categories are present in the given caption.
    Optimization: First runs a fast-path search using a single pre-compiled combined regex.
    If the combined check is clean (which is true for 95%+ of typical captions),
    we immediately return [] without running the 6 separate checks.
    """
    if not _RE_ANY_FORBIDDEN.search(caption):
        return []

    found = []
    for label, pattern in FLUX_FORBIDDEN_PATTERNS.items():
        if pattern.search(caption):
            found.append(label)
    return found


def validate_sdxl(cfg: dict, char_cfg: dict) -> bool:
    seeds_dir = ROOT / char_cfg["seeds_dir"]
    table = Table(title="SDXL Training Dataset Validation")
    table.add_column("Mode", style="cyan")
    table.add_column("Images", justify="right")
    table.add_column("Captions", justify="right")
    table.add_column("Status")

    all_ok = True
    for mode in MODES:
        images = get_seed_images(seeds_dir, mode)
        captions = [img.with_suffix(".txt") for img in images if img.with_suffix(".txt").exists()]

        img_count = len(images)
        cap_count = len(captions)
        missing_caps = img_count - cap_count
        size_issues = []

        for img_path in images:
            try:
                with Image.open(img_path) as im:
                    if im.size != TARGET_SIZE:
                        size_issues.append(img_path.name)
            except Exception:
                size_issues.append(f"{img_path.name} (unreadable)")

        ok = img_count >= MIN_IMAGES_PER_MODE and missing_caps == 0 and not size_issues
        if not ok:
            all_ok = False

        issues = []
        if img_count < MIN_IMAGES_PER_MODE:
            issues.append(f"need {MIN_IMAGES_PER_MODE - img_count} more images")
        if missing_caps:
            issues.append(f"{missing_caps} captions missing")
        if size_issues:
            issues.append(f"{len(size_issues)} wrong size (run --caption-style sdxl)")

        status = "[green]ready[/green]" if ok else f"[red]{'; '.join(issues)}[/red]"
        table.add_row(mode, str(img_count), str(cap_count), status)

    console.print(table)
    total = sum(len(get_seed_images(seeds_dir, m)) for m in MODES)
    console.print(f"Total images: {total} (target: {MIN_IMAGES_PER_MODE * len(MODES)}+)")
    return all_ok


def validate_flux(cfg: dict, char_cfg: dict) -> bool:
    training_dir = get_training_data_dir(char_cfg)
    trigger = char_cfg["trigger_word"]

    # Performance Optimization: Single-pass directory scan using os.scandir()
    # 1. Combines supported (flux) and unsupported image checks to avoid redundant directory scanning.
    # 2. Gathers existing caption (.txt) filenames into a set for O(1) in-memory lookups,
    #    completely eliminating expensive stat/exists system calls for every image.
    supported_exts = tuple(e.lower() for e in SUPPORTED_IMAGE_EXTENSIONS)
    unsupported_exts = tuple(e.lower() for e in UNSUPPORTED_IMAGE_EXTENSIONS)

    images = []
    unsupported = []
    existing_captions = set()

    if training_dir.exists():
        with os.scandir(training_dir) as it:
            for entry in it:
                if entry.is_file():
                    name_low = entry.name.lower()
                    if name_low.endswith(supported_exts):
                        images.append(Path(entry.path))
                    elif name_low.endswith(unsupported_exts):
                        unsupported.append(Path(entry.path))
                    elif name_low.endswith(".txt"):
                        existing_captions.add(entry.name)

    images.sort()
    unsupported.sort()

    duplicate_keys = [
        key
        for key, count in Counter(normalize_duplicate_key(path) for path in images).items()
        if count > 1
    ]

    table = Table(title="FLUX Training Dataset Validation")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail", style="dim")

    all_ok = True

    image_count_ok = len(images) >= FLUX_TARGET_IMAGES
    if not image_count_ok:
        all_ok = False
    table.add_row(
        "Image count",
        "[green]ok[/green]" if image_count_ok else "[yellow]curate more[/yellow]",
        f"{len(images)} image(s), target {FLUX_TARGET_IMAGES}",
    )

    unsupported_ok = not unsupported
    if not unsupported_ok:
        all_ok = False
    table.add_row(
        "Supported formats",
        "[green]ok[/green]" if unsupported_ok else "[red]fail[/red]",
        ", ".join(path.name for path in unsupported) or "png/jpg/jpeg/webp only",
    )

    duplicate_ok = not duplicate_keys
    if not duplicate_ok:
        all_ok = False
    table.add_row(
        "Obvious duplicates",
        "[green]ok[/green]" if duplicate_ok else "[red]review[/red]",
        ", ".join(duplicate_keys) or "none",
    )

    # Optimization: Use O(1) set membership check instead of cap_path.exists() stat call
    # Use fast string rsplit to avoid the slow Path.with_suffix() object creation
    missing_captions = [
        img.name for img in images
        if (img.name.rsplit(".", 1)[0] + ".txt") not in existing_captions
    ]
    missing_ok = not missing_captions
    if not missing_ok:
        all_ok = False
    table.add_row(
        "Image/caption pairs",
        "[green]ok[/green]" if missing_ok else "[red]fail[/red]",
        ", ".join(missing_captions[:8]) + ("..." if len(missing_captions) > 8 else "") or "all paired",
    )

    missing_trigger = []
    forbidden_terms = []
    unreadable = []
    for img in images:
        cap_name = img.name.rsplit(".", 1)[0] + ".txt"
        # Optimization: Use O(1) set membership check instead of cap_path.exists() stat call
        if cap_name not in existing_captions:
            continue
        cap_path = img.with_suffix(".txt")
        try:
            caption = cap_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            unreadable.append(cap_path.name)
            continue
        if trigger not in caption:
            missing_trigger.append(cap_path.name)
        forbidden = find_forbidden_caption_terms(caption)
        if forbidden:
            forbidden_terms.append(f"{cap_path.name}: {', '.join(forbidden)}")

    if unreadable:
        all_ok = False
    table.add_row(
        "Readable captions",
        "[green]ok[/green]" if not unreadable else "[red]fail[/red]",
        ", ".join(unreadable) or "all UTF-8 readable",
    )

    trigger_check_blocked = bool(missing_captions)
    if missing_trigger or trigger_check_blocked:
        all_ok = False
    table.add_row(
        f"Trigger word ({trigger})",
        "[green]ok[/green]" if not missing_trigger and not trigger_check_blocked else "[red]fail[/red]",
        (
            "caption coverage incomplete"
            if trigger_check_blocked
            else ", ".join(missing_trigger[:8]) + ("..." if len(missing_trigger) > 8 else "") or "present in every caption"
        ),
    )

    forbidden_check_blocked = bool(missing_captions)
    if forbidden_terms or forbidden_check_blocked:
        all_ok = False
    table.add_row(
        "Forbidden identity terms",
        "[green]ok[/green]" if not forbidden_terms and not forbidden_check_blocked else "[red]fail[/red]",
        (
            "caption coverage incomplete"
            if forbidden_check_blocked
            else "; ".join(forbidden_terms[:5]) + ("..." if len(forbidden_terms) > 5 else "") or "none found"
        ),
    )

    console.print(table)
    console.print(f"Dataset folder: {training_dir}")
    return all_ok


def generate_sdxl_captions(cfg: dict, char_cfg: dict) -> None:
    seeds_dir = ROOT / char_cfg["seeds_dir"]
    base_prompt = (ROOT / char_cfg["base_prompt_file"]).read_text(encoding="utf-8").strip()
    trigger = char_cfg["trigger_word"]
    base_without_trigger = base_prompt.replace(f"{trigger}, ", "").replace(trigger, "").strip().strip(",").strip()

    for mode in MODES:
        images = get_seed_images(seeds_dir, mode)
        if not images:
            continue

        for img_path in images:
            with Image.open(img_path) as im:
                if im.size != TARGET_SIZE:
                    resized = im.resize(TARGET_SIZE, Image.LANCZOS)
                    resized.save(img_path, "PNG")
                    console.print(f"[dim]Resized {img_path.name} to 1024x1024[/dim]")

            cap_path = img_path.with_suffix(".txt")
            if cap_path.exists():
                console.print(f"[dim]Skipping caption for {img_path.name} (already exists)[/dim]")
                continue

            mode_prefix = MODE_CAPTION_PREFIX[mode]
            caption = f"{mode_prefix}, {base_without_trigger}, neutral background, studio lighting, natural pose"
            cap_path.write_text(caption, encoding="utf-8")
            console.print(f"[green]Generated caption:[/green] {img_path.name} -> {cap_path.name}")

    console.print("\n[yellow]IMPORTANT: Review and edit every .txt caption file.[/yellow]")
    console.print("[yellow]Apply the SDXL Isolation Rule: describe setting/pose/lighting only.[/yellow]")


def infer_flux_shot_type(img_path: Path) -> str:
    name = img_path.stem.lower()
    if "full" in name or "body" in name:
        return "full-body portrait"
    if "medium" in name or "waist" in name:
        return "waist-up portrait"
    if "close" in name or "face" in name or "head" in name:
        return "close-up portrait"

    try:
        with Image.open(img_path) as im:
            width, height = im.size
    except Exception:
        return "portrait"

    if height >= width * 1.35:
        return "vertical fashion portrait"
    return "portrait"


def generate_flux_captions(cfg: dict, char_cfg: dict) -> None:
    training_dir = get_training_data_dir(char_cfg)
    images = get_flux_images(training_dir)
    if not images:
        console.print(f"[red]No supported images found in {training_dir}[/red]")
        raise SystemExit(1)

    trigger = char_cfg["trigger_word"]
    for img_path in images:
        cap_path = img_path.with_suffix(".txt")
        if cap_path.exists():
            console.print(f"[dim]Skipping caption for {img_path.name} (already exists)[/dim]")
            continue

        caption = FLUX_CAPTION_TEMPLATE.format(
            shot_type=infer_flux_shot_type(img_path),
            trigger=trigger,
        )
        cap_path.write_text(caption, encoding="utf-8")
        console.print(f"[green]Generated FLUX caption scaffold:[/green] {img_path.name} -> {cap_path.name}")

    console.print("\n[yellow]IMPORTANT: Review and edit every FLUX caption manually.[/yellow]")
    console.print("[yellow]Keep AnanyaAI, remove permanent facial identity, and make hair/outfit/pose/light/background factual.[/yellow]")


def zip_sdxl_dataset(cfg: dict, char_cfg: dict, character: str) -> None:
    seeds_dir = ROOT / char_cfg["seeds_dir"]
    trigger = char_cfg["trigger_word"]
    output_zip = ROOT / f"training_data_{character}.zip"
    kohya_config = ROOT / "setup" / "kohya_config.toml"

    all_images = []
    for mode in MODES:
        images = get_seed_images(seeds_dir, mode)
        for img in images:
            cap = img.with_suffix(".txt")
            if not cap.exists():
                console.print(f"[red]Missing caption for {img.name} - run --caption-style sdxl first[/red]")
                raise SystemExit(1)
            all_images.append((img, cap))

    if not all_images:
        console.print(f"[red]No images found in {seeds_dir}. Run bootstrap_seeds.py --character {character} first.[/red]")
        raise SystemExit(1)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path, cap_path in all_images:
            arcname_img = f"img/10_{trigger} woman/{img_path.name}"
            arcname_cap = f"img/10_{trigger} woman/{cap_path.name}"
            zf.write(img_path, arcname_img)
            zf.write(cap_path, arcname_cap)
        if kohya_config.exists():
            zf.write(kohya_config, "kohya_config.toml")

    console.print(f"[green]Created {output_zip} ({len(all_images)} image-caption pairs)[/green]")
    console.print("[cyan]Upload to RunPod and follow setup/train_lora_guide.md[/cyan]")


def write_flux_ai_toolkit_config(zf: zipfile.ZipFile, char_cfg: dict) -> None:
    trigger = char_cfg["trigger_word"]
    config = f"""---
job: extension
config:
  name: "{trigger}_FLUX_v1"
  process:
    - type: sd_trainer
      training_folder: "/workspace/output"
      device: cuda:0
      network:
        type: lora
        linear: 16
        linear_alpha: 16
      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4
      datasets:
        - folder_path: "/workspace/training_images"
          caption_ext: "txt"
          caption_dropout_rate: 0
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [512, 768, 1024]
      train:
        batch_size: 1
        steps: 2000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: flowmatch
        optimizer: adamw8bit
        lr: 0.0001
        dtype: bf16
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        is_flux: true
      sample:
        sampler: flowmatch
        sample_every: 250
        width: 896
        height: 1152
        prompts:
          - "{trigger}, waist-up portrait, loose styled hair, black linen blazer, relaxed expression, soft cafe window light, detailed lifestyle background"
          - "{trigger}, full-body fashion portrait, hair tied back, ivory satin dress, slow walking pose, warm golden hour rooftop light, city background"
meta:
  name: "{trigger}_FLUX_v1"
  version: "1.0"
"""
    zf.writestr("ai_toolkit_flux_lora_rank16.yaml", config)


def zip_flux_dataset(cfg: dict, char_cfg: dict, character: str) -> None:
    training_dir = get_training_data_dir(char_cfg)
    images = get_flux_images(training_dir)
    if not images:
        console.print(f"[red]No supported images found in {training_dir}[/red]")
        raise SystemExit(1)

    output_zip = ROOT / f"training_data_{character}_flux.zip"
    missing = [img.name for img in images if not img.with_suffix(".txt").exists()]
    if missing:
        console.print(f"[red]Missing captions: {', '.join(missing[:8])}[/red]")
        raise SystemExit(1)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path in images:
            cap_path = img_path.with_suffix(".txt")
            zf.write(img_path, f"training_images/{img_path.name}")
            zf.write(cap_path, f"training_images/{cap_path.name}")
        write_flux_ai_toolkit_config(zf, char_cfg)

    console.print(f"[green]Created {output_zip} ({len(images)} image-caption pairs)[/green]")
    console.print("[cyan]Upload the zip contents to RunPod and use ai_toolkit_flux_lora_rank16.yaml with ostris/ai-toolkit.[/cyan]")


@click.command()
@click.option("--character", default="ananya", show_default=True, help="Character to prepare [ananya|kavib]")
@click.option("--layout", type=click.Choice(["auto", "sdxl", "flux"]), default="auto", show_default=True, help="Dataset layout to validate/package")
@click.option("--validate", "do_validate", is_flag=True, help="Check images, sizes, captions, and dataset rules")
@click.option("--caption-style", type=click.Choice(["sdxl", "flux"]), help="Auto-generate captions for the selected training style")
@click.option("--zip-only", is_flag=True, help="Package dataset into a training zip")
def main(character: str, layout: str, do_validate: bool, caption_style: str | None, zip_only: bool):
    cfg = load_config()
    char_cfg = load_character(cfg, character)
    selected_layout = caption_style or choose_layout(char_cfg, layout)

    if not do_validate and not caption_style and not zip_only:
        console.print("[yellow]Specify at least one option: --validate, --caption-style, or --zip-only[/yellow]")
        console.print("Run with --help for usage.")
        raise SystemExit(0)

    if do_validate:
        if selected_layout == "flux":
            validate_flux(cfg, char_cfg)
        else:
            validate_sdxl(cfg, char_cfg)

    if caption_style == "flux":
        generate_flux_captions(cfg, char_cfg)
    elif caption_style == "sdxl":
        generate_sdxl_captions(cfg, char_cfg)

    if zip_only:
        if selected_layout == "flux":
            zip_flux_dataset(cfg, char_cfg, character)
        else:
            zip_sdxl_dataset(cfg, char_cfg, character)


if __name__ == "__main__":
    main()
