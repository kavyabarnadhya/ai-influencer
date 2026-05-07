import sys
from datetime import datetime
from pathlib import Path

import click
import yaml
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from comfyui_api import ComfyUIClient, ComfyUIError, inject_workflow_values, load_workflow, find_comfyui_port

console = Console()
ROOT = Path(__file__).parent.parent


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--image", required=True, help="Input photo to extract pose from")
@click.option("--output", default=None, help="Output directory for skeleton PNG (default: same dir as input)")
@click.option("--prefix", default=None, help="Output filename prefix (default: pose_<input_stem>)")
def main(image: str, output: str | None, prefix: str | None):
    """Extract OpenPose skeleton from a photo using SDPose. Output PNG can be used as --pose ref in generate.py."""
    cfg = load_config()
    comfy_cfg = cfg["comfyui"]

    host = comfy_cfg["host"]
    port = find_comfyui_port(host, [comfy_cfg["port"], 8000, 8188, 8002])
    if port is None:
        console.print("[red]ComfyUI is not running. Start it first.[/red]")
        raise SystemExit(1)
    if port != comfy_cfg["port"]:
        console.print(f"[yellow]ComfyUI found on port {port}[/yellow]")

    client = ComfyUIClient(host, port)

    img_path = Path(image).resolve()
    if not img_path.exists():
        console.print(f"[red]Image not found: {image}[/red]")
        raise SystemExit(1)

    out_dir = Path(output).resolve() if output else img_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = prefix or f"pose_{img_path.stem}"

    workflow_path = ROOT / cfg["paths"]["workflows_dir"] / "sdpose_extract.json"
    if not workflow_path.exists():
        console.print(f"[red]Workflow not found: {workflow_path}[/red]")
        raise SystemExit(1)

    console.print(f"[dim]Uploading image: {img_path.name}...[/dim]")
    uploaded = client.upload_image(str(img_path))

    workflow_data = load_workflow(str(workflow_path))
    overrides = {
        "_claude_inject_input_image": {"inputs.image": uploaded},
        "_claude_inject_output_prefix": {"inputs.filename_prefix": file_prefix},
    }
    patched = inject_workflow_values(workflow_data, overrides)

    console.print("[cyan]Extracting pose...[/cyan]")
    try:
        prompt_id = client.submit_workflow(patched)
        images = client.wait_for_completion(prompt_id, timeout=comfy_cfg["timeout"])
    except ComfyUIError as e:
        console.print(f"[red]Extraction failed: {e}[/red]")
        raise SystemExit(1)

    for img_meta in images:
        img_bytes = client.download_image(
            img_meta["filename"], img_meta.get("subfolder", ""), img_meta.get("type", "output")
        )
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"{file_prefix}_{timestamp}.png"
        dest = out_dir / filename
        dest.write_bytes(img_bytes)
        console.print(f"[green]Saved skeleton:[/green] {dest}")
        console.print(f"[dim]Use with: generate.py --pose \"{dest}\" --workflow t2i_sdxl_lora_ipadapter_controlnet[/dim]")


if __name__ == "__main__":
    main()
