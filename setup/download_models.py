import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console(highlight=False)

MODELS = {
    "6gb": [
        {
            "name": "Juggernaut XL v9 (RunDiffusion)",
            "repo": "RunDiffusion/Juggernaut-XL-v9",
            "filename": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
            "subfolder": None,
            "dest_subdir": "checkpoints",
        },
        {
            "name": "IP-Adapter Plus Face SDXL",
            "repo": "h94/IP-Adapter",
            "filename": "ip-adapter-plus-face_sdxl_vit-h.safetensors",
            "subfolder": "sdxl_models",
            "dest_subdir": "ipadapter",
        },
        {
            "name": "FLUX.1-schnell Q4_K_S GGUF",
            "repo": "city96/FLUX.1-schnell-gguf",
            "filename": "flux1-schnell-Q4_K_S.gguf",
            "subfolder": None,
            "dest_subdir": "unet",
        },
        {
            "name": "FLUX.1-schnell Q3_K_S GGUF (low-VRAM fallback)",
            "repo": "city96/FLUX.1-schnell-gguf",
            "filename": "flux1-schnell-Q3_K_S.gguf",
            "subfolder": None,
            "dest_subdir": "unet",
        },
        {
            "name": "FLUX VAE",
            "repo": "black-forest-labs/FLUX.1-schnell",
            "filename": "ae.safetensors",
            "subfolder": None,
            "dest_subdir": "vae",
        },
        {
            "name": "CLIP-L",
            "repo": "comfyanonymous/flux_text_encoders",
            "filename": "clip_l.safetensors",
            "subfolder": None,
            "dest_subdir": "clip",
        },
        {
            "name": "T5-XXL fp8",
            "repo": "comfyanonymous/flux_text_encoders",
            "filename": "t5xxl_fp8_e4m3fn.safetensors",
            "subfolder": None,
            "dest_subdir": "clip",
        },
    ]
}
MODELS["8gb"] = [m for m in MODELS["6gb"] if "Q3" not in m["name"]]
MODELS["12gb+"] = MODELS["8gb"]


def _download_yolo_models(models_root: Path) -> None:
    """Download YOLO models used by skin_color_match.py. Source: ultralytics auto-download from GitHub releases."""
    seg_dest = models_root / "ultralytics" / "segm" / "yolov8n-seg.pt"
    if seg_dest.exists():
        console.print(f"[dim]Skipping YOLOv8n-seg (already present at {seg_dest})[/dim]")
        return

    console.print("[cyan]Downloading YOLOv8n-seg (person segmentation, ~6MB)...[/cyan]")
    try:
        from ultralytics import YOLO
        import shutil
        seg_dest.parent.mkdir(parents=True, exist_ok=True)
        # ultralytics auto-downloads to CWD on first instantiation; move to configured path
        YOLO("yolov8n-seg.pt")
        cwd_path = Path("yolov8n-seg.pt")
        if cwd_path.exists():
            shutil.move(str(cwd_path), str(seg_dest))
            console.print(f"[green]OK: YOLOv8n-seg -> {seg_dest}[/green]")
        else:
            console.print("[red]FAILED: YOLOv8n-seg download produced no file in CWD[/red]")
    except ImportError:
        console.print("[red]FAILED: ultralytics package not installed. Run: pip install ultralytics[/red]")
    except Exception as e:
        short_err = str(e).split("\n")[0][:120]
        console.print(f"[red]FAILED: YOLOv8n-seg: {short_err}[/red]")


@click.command()
@click.option("--hf-token", required=True, envvar="HF_TOKEN", help="Hugging Face access token (or set HF_TOKEN env var)")
@click.option("--vram", type=click.Choice(["6gb", "8gb", "12gb+"]), default="6gb", show_default=True, help="VRAM tier")
@click.option("--comfyui-path", default="C:/ComfyUI", show_default=True, help="Path to ComfyUI user data directory")
def main(hf_token: str, vram: str, comfyui_path: str):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        console.print("huggingface-hub not installed. Run: pip install huggingface-hub")
        raise SystemExit(1)

    models_root = Path(comfyui_path) / "models"
    model_list = MODELS[vram]

    table = Table(title=f"Models to download ({vram} profile - {len(model_list)} files)")
    table.add_column("Model", style="cyan")
    table.add_column("Destination")
    for m in model_list:
        dest = models_root / m["dest_subdir"] / m["filename"]
        status = "[green]already present[/green]" if dest.exists() else "[yellow]will download[/yellow]"
        table.add_row(m["name"], f"{dest.parent} {status}")
    console.print(table)

    failed = []
    for m in model_list:
        dest_dir = models_root / m["dest_subdir"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / m["filename"]

        if dest.exists():
            console.print(f"[dim]Skipping {m['name']} (already present)[/dim]")
            continue

        console.print(f"[cyan]Downloading {m['name']}...[/cyan]")
        try:
            kwargs = {
                "repo_id": m["repo"],
                "filename": m["filename"],
                "token": hf_token,
                "local_dir": str(dest_dir),
            }
            if m["subfolder"]:
                kwargs["subfolder"] = m["subfolder"]

            hf_hub_download(**kwargs)
            console.print(f"[green]OK: {m['name']}[/green]")
        except Exception as e:
            short_err = str(e).split("\n")[0][:120]
            console.print(f"[red]FAILED: {m['name']}: {short_err}[/red]")
            failed.append(m["name"])

    if failed:
        console.print(f"\n[yellow]Failed downloads ({len(failed)}): {', '.join(failed)}[/yellow]")
        console.print("[yellow]Check that your HF token has access and you accepted model licenses on huggingface.co[/yellow]")
    else:
        console.print("\n[bold green]All HF downloads complete.[/bold green]")

    # YOLO models for skin_color_match.py (separate source — GitHub releases via ultralytics)
    _download_yolo_models(models_root)

    console.print("Next: python setup/verify_setup.py")


if __name__ == "__main__":
    main()
