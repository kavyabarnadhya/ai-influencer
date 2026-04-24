#!/usr/bin/env python3
"""Download all ComfyUI models via huggingface-hub.

Each model entry pins the exact filename and optionally a SHA256 hash.
If sha256 is set, the file is verified after download and on skip-existing checks.

Usage:
    python setup/download_models.py --hf-token YOUR_TOKEN
    python setup/download_models.py --hf-token YOUR_TOKEN --vram 6
    python setup/download_models.py --hf-token YOUR_TOKEN --no-skip-existing
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm
    from rich.console import Console
    from rich.table import Table
except ImportError:
    print("Missing dependencies. Run: pip install huggingface-hub tqdm rich")
    sys.exit(1)

console = Console()

# ---------------------------------------------------------------------------
# Model manifest — exact filenames + optional SHA256 for reproducibility.
# sha256 values: fill in after first successful download via sha256sum <file>.
# ---------------------------------------------------------------------------
MODELS = [
    {
        "name": "Juggernaut XL v10 (SDXL checkpoint)",
        "repo": "RunDiffusion/Juggernaut-X-v10",
        "filename": "juggernautXL_v10RunDiffusion.safetensors",
        "dest_subdir": "checkpoints",
        "sha256": None,
        "vram_min": 0,
        "note": "Primary base model. Verify commercial license before monetizing.",
    },
    {
        "name": "IP-Adapter Plus Face SDXL",
        "repo": "h94/IP-Adapter",
        "filename": "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin",
        "dest_filename": "ip-adapter-plus-face_sdxl_vit-h.bin",
        "dest_subdir": "ipadapter",
        "sha256": None,
        "vram_min": 0,
    },
    {
        "name": "FLUX.1-schnell GGUF Q4_K_S (UNet ~6.2GB)",
        "repo": "city96/FLUX.1-schnell-gguf",
        "filename": "flux1-schnell-Q4_K_S.gguf",
        "dest_subdir": "unet",
        "sha256": None,
        "vram_min": 0,
        "note": "UNet marginally exceeds 6GB VRAM. T5+VAE offload to RAM required. "
                "Fall back to flux1-schnell-Q3_K_S.gguf (~4.7GB) if OOM.",
    },
    {
        "name": "FLUX VAE (ae.safetensors)",
        "repo": "black-forest-labs/FLUX.1-schnell",
        "filename": "ae.safetensors",
        "dest_subdir": "vae",
        "sha256": None,
        "vram_min": 0,
    },
    {
        "name": "FLUX CLIP-L",
        "repo": "black-forest-labs/FLUX.1-schnell",
        "filename": "clip_l.safetensors",
        "dest_subdir": "clip",
        "sha256": None,
        "vram_min": 0,
    },
    {
        "name": "FLUX T5-XXL fp8",
        "repo": "black-forest-labs/FLUX.1-schnell",
        "filename": "t5xxl_fp8_e4m3fn.safetensors",
        "dest_subdir": "clip",
        "sha256": None,
        "vram_min": 0,
    },
    {
        "name": "YOLOv8 face detector (FaceDetailer)",
        "repo": "Ultralytics/assets",
        "filename": "face_yolov8n.pt",
        "dest_subdir": "ultralytics",
        "sha256": None,
        "vram_min": 0,
        "note": "Used by Impact Pack FaceDetailer — no insightface required.",
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(model: dict, models_dir: Path, token: str, skip_existing: bool) -> bool:
    dest_subdir = models_dir / model["dest_subdir"]
    dest_subdir.mkdir(parents=True, exist_ok=True)

    dest_name = model.get("dest_filename") or Path(model["filename"]).name
    dest_path = dest_subdir / dest_name
    expected_hash = model.get("sha256")

    if skip_existing and dest_path.exists():
        if expected_hash:
            actual = sha256_file(dest_path)
            if actual == expected_hash:
                console.print(f"  [dim]SKIP[/dim] {dest_name} (present, hash verified)")
                return True
            else:
                console.print(f"  [yellow]HASH MISMATCH[/yellow] {dest_name} — re-downloading")
        else:
            console.print(f"  [dim]SKIP[/dim] {dest_name} (present, no hash to verify)")
            return True

    console.print(f"  [cyan]DOWNLOAD[/cyan] {model['name']}")
    if model.get("note"):
        console.print(f"  [dim]  Note: {model['note']}[/dim]")

    try:
        downloaded = hf_hub_download(
            repo_id=model["repo"],
            filename=model["filename"],
            token=token,
            local_dir=str(dest_subdir),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may place file in a cache subfolder; move if needed
        downloaded_path = Path(downloaded)
        if downloaded_path != dest_path and downloaded_path.exists():
            downloaded_path.rename(dest_path)
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "credentials" in msg.lower() or "token" in msg.lower():
            console.print(
                f"\n  [red]AUTH ERROR[/red] for {model['repo']}\n"
                "  Fix: pass --hf-token YOUR_TOKEN\n"
                "  Get a token at: https://huggingface.co/settings/tokens\n"
                "  Some repos (FLUX) require accepting the model license on the HF page first."
            )
        else:
            console.print(f"  [red]FAILED[/red] {model['name']}: {exc}")
        return False

    if expected_hash:
        actual = sha256_file(dest_path)
        if actual != expected_hash:
            console.print(
                f"  [red]SHA256 MISMATCH[/red] {dest_name}\n"
                f"    expected: {expected_hash}\n"
                f"    actual:   {actual}"
            )
            return False
        console.print(f"  [green]OK[/green] {dest_name} (hash verified)")
    else:
        console.print(f"  [green]OK[/green] {dest_name}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Download ComfyUI models for ai-influencer pipeline")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""), help="HuggingFace API token")
    parser.add_argument("--models-dir", default="C:/ComfyUI/models", help="ComfyUI models directory")
    parser.add_argument("--vram", type=int, default=6, help="VRAM in GB (filters large models)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", default=True)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    console.print(f"\n[bold]AI-Influencer Model Downloader[/bold]")
    console.print(f"Models dir : {models_dir}")
    console.print(f"VRAM       : {args.vram}GB")
    console.print(f"Skip exist : {args.skip_existing}\n")

    if not args.hf_token:
        console.print(
            "[yellow]WARNING[/yellow]: No --hf-token provided. "
            "Private/gated repos (FLUX, Juggernaut) will fail with 401.\n"
            "Get a token at https://huggingface.co/settings/tokens\n"
        )

    results = []
    for model in MODELS:
        ok = download_model(model, models_dir, args.hf_token, args.skip_existing)
        results.append((model["name"], ok))

    console.print("\n[bold]Summary[/bold]")
    all_ok = True
    for name, ok in results:
        status = "[green]OK[/green]" if ok else "[red]FAILED[/red]"
        console.print(f"  {status}  {name}")
        if not ok:
            all_ok = False

    if not all_ok:
        console.print(
            "\n[yellow]Some downloads failed.[/yellow] "
            "Fix auth/network issues and re-run — --skip-existing will resume safely."
        )
        sys.exit(1)

    console.print(
        "\n[bold green]All models downloaded.[/bold green]\n"
        "Next step: follow setup/train_lora_guide.md to train your character LoRA,\n"
        "then place KaviB_v1_Prod.safetensors in:\n"
        f"  {models_dir / 'loras' / 'KaviB_v1_Prod.safetensors'}\n"
        "Then run: python setup/verify_setup.py"
    )


if __name__ == "__main__":
    main()
