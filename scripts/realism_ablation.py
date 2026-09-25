"""One-factor-at-a-time SDXL/LoRA and ReActor realism comparison.

Run against local ComfyUI; requires the repo's SDXL, FaceDetailer and ReActor
workflows/models. No generated image is posted or used for training.
"""

import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import yaml
from PIL import Image, ImageDraw, ImageFont

from comfyui_api import ComfyUIClient, ComfyUIError, find_comfyui_port, inject_workflow_values, load_workflow

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Case:
    label: str
    swap: bool = True
    restore: float = 1.0
    restore_model: str = "codeformer-v0.1.0.pth"
    lora: float = 0.9
    cfg: float = 8.0


def cases() -> list[Case]:
    """Baseline plus nine single-change variants. No confounded factorial cells."""
    return [
        Case("baseline | swap ON | CodeFormer 1 | LoRA .9 | CFG 8"),
        Case("swap OFF", swap=False),
        Case("CodeFormer .5", restore=0.5),
        Case("CodeFormer OFF", restore_model="none"),
        Case("LoRA .65", lora=0.65),
        Case("LoRA .75", lora=0.75),
        Case("LoRA .85", lora=0.85),
        Case("CFG 5", cfg=5),
        Case("CFG 6", cfg=6),
        Case("CFG 7", cfg=7),
    ]


def generation_workflow(template: dict, config: dict, prompt: str, seed: int, case: Case,
                         width: int | None = None, height: int | None = None) -> dict:
    gen = config["generation"]
    char = config["characters"]["ananya"]
    wf = inject_workflow_values(template, {
        "_claude_inject_checkpoint": {"inputs.ckpt_name": config["models"]["checkpoint"]},
        "_claude_inject_lora": {
            "inputs.lora_name": char["lora"],
            "inputs.strength_model": case.lora,
            "inputs.strength_clip": case.lora,
        },
        "_claude_inject_prompt": {"inputs.text": prompt},
        "_claude_inject_negative": {"inputs.text": gen["negative_prompt"]},
        "_claude_inject_latent": {"inputs.width": width or gen["width"], "inputs.height": height or gen["height"]},
        "_claude_inject_seed": {"inputs.seed": seed, "inputs.steps": gen["steps"], "inputs.cfg": case.cfg},
    })
    # Face and hand detailing are secondary samplers. Hold their seed, CFG and
    # checkpoint constant except for the axis under test, or they confound it.
    for node_id, node in list(wf.items()):
        if isinstance(node, dict) and node.get("class_type") == "FaceDetailer":
            wf[node_id] = {**node, "inputs": {**node["inputs"], "seed": seed, "cfg": case.cfg}}
    return wf


def swap_workflow(template: dict, face_name: str, target_name: str, restore: float, restore_model: str) -> dict:
    wf = inject_workflow_values(template, {
        "_claude_inject_source_image": {"inputs.image": face_name},
        "_claude_inject_target_image": {"inputs.image": target_name},
        "_claude_reactor_swap": {
            "inputs.face_restore_visibility": restore,
            "inputs.face_restore_model": restore_model,
        },
    })
    return wf


def render_sheet(images: list[tuple[Case, Path]], destination: Path,
                 crop: tuple[int, int, int, int] | None = None) -> None:
    """Paste original pixels, never resample or silently crop different regions."""
    tiles = []
    for case, path in images:
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            if crop is not None:
                x, y, w, h = crop
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > rgb.width or y + h > rgb.height:
                    raise ValueError(f"Crop {crop} falls outside {path.name} ({rgb.width}x{rgb.height})")
                rgb = rgb.crop((x, y, x + w, y + h))
            tiles.append((case.label, rgb))
    if not tiles:
        raise ValueError("No images to compare")
    sizes = {image.size for _, image in tiles}
    if len(sizes) != 1:
        raise ValueError(f"Outputs differ in size; cannot compare at native scale: {sizes}")
    w, h = tiles[0][1].size
    columns = min(3, len(tiles))
    label_h = 54
    canvas = Image.new("RGB", (columns * w, ((len(tiles) + columns - 1) // columns) * (h + label_h)), "#181818")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 18) if Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf").exists() else ImageFont.load_default()
    for index, (label, image) in enumerate(tiles):
        x, y = (index % columns) * w, (index // columns) * (h + label_h)
        draw.text((x + 12, y + 13), label, fill="white", font=font)
        canvas.paste(image, (x, y + label_h))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def run_image(client: ComfyUIClient, workflow: dict, timeout: int) -> bytes:
    prompt_id = client.submit_workflow(workflow)
    outputs = client.wait_for_completion(prompt_id, timeout=timeout)
    if len(outputs) != 1:
        raise ComfyUIError(f"Expected one image from {prompt_id}, got {len(outputs)}")
    ref = outputs[0]
    return client.download_image(ref["filename"], ref.get("subfolder", ""), ref.get("type", "output"))


@click.command()
@click.option("--face-ref", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="One source face image used for every swapped cell")
@click.option("--seed", type=int, required=True, help="Fixed nonnegative seed for every generated cell")
@click.option("--prompt", required=True, help="Fixed scene/outfit prompt (AnanyaAI is prepended if absent)")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=ROOT / "output" / "realism_ablation")
@click.option("--crop", nargs=4, type=int, default=None, help="Optional same native-pixel crop: X Y WIDTH HEIGHT")
@click.option("--port", type=int, default=None, help="ComfyUI port; auto-detect if omitted")
@click.option("--width", type=int, default=None, help="Override config.yaml generation width (e.g. reels canvas)")
@click.option("--height", type=int, default=None, help="Override config.yaml generation height (e.g. reels canvas)")
def main(face_ref: Path, seed: int, prompt: str, output_dir: Path,
         crop: tuple[int, int, int, int] | None, port: int | None,
         width: int | None, height: int | None) -> None:
    if seed < 0 or seed >= 2**64:
        raise click.BadParameter("Seed must be in [0, 2**64)", param_hint="--seed")
    if not prompt.strip():
        raise click.BadParameter("Provide a scene/outfit prompt", param_hint="--prompt")
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    gen_template = load_workflow(str(ROOT / "workflows" / "t2i_sdxl_lora.json"))
    swap_template = load_workflow(str(ROOT / "workflows" / "faceswap_reactor.json"))
    port = port or find_comfyui_port(config["comfyui"]["host"])
    if port is None:
        raise click.ClickException("ComfyUI not reachable. Start it first or specify --port.")
    client = ComfyUIClient(config["comfyui"]["host"], port)
    full_prompt = prompt.strip()
    trigger = config["characters"]["ananya"]["trigger_word"]
    if trigger.lower() not in full_prompt.lower():
        full_prompt = f"{trigger}, {full_prompt}"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        source = client.upload_image(str(face_ref))
        results = []
        for index, case in enumerate(cases()):
            click.echo(f"[{index + 1}/{len(cases())}] {case.label}")
            generated = run_image(client, generation_workflow(gen_template, config, full_prompt, seed, case, width, height),
                                  config["comfyui"]["timeout"])
            if case.swap:
                # Unique upload name per generated image prevents remote-name collisions.
                target = client.upload_image_data(generated, f"ablation_{seed}_{index:02d}.png")
                final = run_image(client, swap_workflow(swap_template, source, target, case.restore, case.restore_model),
                                  config["comfyui"]["timeout"])
            else:
                final = generated
            with Image.open(io.BytesIO(final)) as check:
                check.verify()
            path = output_dir / f"{index:02d}_{'swap' if case.swap else 'raw'}.png"
            path.write_bytes(final)
            results.append((case, path))
        render_sheet(results, output_dir / "contact_sheet.png", crop)
        (output_dir / "manifest.json").write_text(json.dumps({
            "seed": seed, "prompt": full_prompt, "face_ref": str(face_ref),
            "checkpoint": config["models"]["checkpoint"], "steps": config["generation"]["steps"],
            "width": width or config["generation"]["width"], "height": height or config["generation"]["height"],
            "crop": crop, "postprocessing": "none (no skin_color_match or UltraSharp)",
            "cases": [{**asdict(case), "file": path.name} for case, path in results],
        }, indent=2), encoding="utf-8")
        click.echo(f"Saved {len(results)} images and {output_dir / 'contact_sheet.png'}")
    except (ComfyUIError, ValueError, OSError) as exc:
        raise click.ClickException(f"Ablation stopped; completed cells remain in {output_dir}: {exc}") from exc


if __name__ == "__main__":
    main()
