import shutil
import zipfile
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

MODE_CAPTION_PREFIX = {
    "closeup": "close-up face shot, head and shoulders",
    "medium": "waist-up shot, medium portrait",
    "fullbody": "full body shot, full length portrait",
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


def get_seed_images(seeds_dir: Path, mode: str) -> list[Path]:
    mode_dir = seeds_dir / mode
    if not mode_dir.exists():
        return []
    return sorted(mode_dir.glob("*.png"))


def validate(cfg: dict, char_cfg: dict) -> bool:
    seeds_dir = ROOT / char_cfg["seeds_dir"]
    table = Table(title="Training Dataset Validation")
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
            issues.append(f"{len(size_issues)} wrong size (run --caption-style to resize)")

        status = "[green]ready[/green]" if ok else f"[red]{'; '.join(issues)}[/red]"
        table.add_row(mode, str(img_count), str(cap_count), status)

    console.print(table)
    total = sum(len(get_seed_images(seeds_dir, m)) for m in MODES)
    console.print(f"Total images: {total} (target: {MIN_IMAGES_PER_MODE * len(MODES)}+)")
    return all_ok


def generate_captions(cfg: dict, char_cfg: dict, caption_style: str) -> None:
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
            if caption_style == "sdxl":
                caption = f"{mode_prefix}, {base_without_trigger}, neutral background, studio lighting, natural pose"
            else:
                caption = f"{mode_prefix}, photorealistic portrait, {base_without_trigger}"

            cap_path.write_text(caption, encoding="utf-8")
            console.print(f"[green]Generated caption:[/green] {img_path.name} -> {cap_path.name}")

    console.print("\n[yellow]IMPORTANT: Review and edit every .txt caption file.[/yellow]")
    console.print("[yellow]Apply the Isolation Rule — describe setting/pose/lighting only.[/yellow]")
    console.print("[yellow]Remove any physical descriptors (face, hair, skin, eyes).[/yellow]")


def zip_dataset(cfg: dict, char_cfg: dict, character: str) -> None:
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
                console.print(f"[red]Missing caption for {img.name} — run --caption-style first[/red]")
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


@click.command()
@click.option("--character", default="ananya", show_default=True, help="Character to prepare [ananya|kavib]")
@click.option("--validate", "do_validate", is_flag=True, help="Check images, sizes, and caption coverage")
@click.option("--caption-style", type=click.Choice(["sdxl", "flux"]), help="Auto-generate captions and resize images")
@click.option("--zip-only", is_flag=True, help="Package dataset into training_data_{character}.zip")
def main(character: str, do_validate: bool, caption_style: str | None, zip_only: bool):
    cfg = load_config()
    char_cfg = load_character(cfg, character)

    if not do_validate and not caption_style and not zip_only:
        console.print("[yellow]Specify at least one option: --validate, --caption-style, or --zip-only[/yellow]")
        console.print("Run with --help for usage.")
        raise SystemExit(0)

    if do_validate:
        validate(cfg, char_cfg)

    if caption_style:
        generate_captions(cfg, char_cfg, caption_style)

    if zip_only:
        zip_dataset(cfg, char_cfg, character)


if __name__ == "__main__":
    main()
