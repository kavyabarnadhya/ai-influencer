"""Apply ReActor faceswap (Ananya identity lock) onto an already-generated image.

Uses the winning config from the realism ablation (PR #158): face_restore_model="none"
gives natural skin texture; CodeFormer restore smooths skin toward plastic. See
output/realism_ablation_v2/RESULTS.md for the visual evidence.

Usage:
    python scripts/apply_faceswap.py --face-ref PATH --target PATH --output PATH
    python scripts/apply_faceswap.py --face-ref PATH --target PATH --output PATH --restore-model codeformer-v0.1.0.pth
"""

from pathlib import Path

import click

from comfyui_api import ComfyUIClient, ComfyUIError, find_comfyui_port, inject_workflow_values, load_workflow

ROOT = Path(__file__).resolve().parent.parent


@click.command()
@click.option("--face-ref", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--target", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--restore-model", default="none", show_default=True,
              help='ReActor face_restore_model. "none" = winning ablation config (natural skin).')
@click.option("--restore-visibility", type=float, default=1.0, show_default=True)
@click.option("--port", type=int, default=None)
def main(face_ref: Path, target: Path, output: Path, restore_model: str, restore_visibility: float, port: int | None):
    port = port or find_comfyui_port()
    if port is None:
        raise click.ClickException("ComfyUI not reachable. Start it first or specify --port.")
    client = ComfyUIClient("127.0.0.1", port)
    template = load_workflow(str(ROOT / "workflows" / "faceswap_reactor.json"))

    source_name = client.upload_image(str(face_ref))
    target_name = client.upload_image(str(target))
    wf = inject_workflow_values(template, {
        "_claude_inject_source_image": {"inputs.image": source_name},
        "_claude_inject_target_image": {"inputs.image": target_name},
        "_claude_reactor_swap": {
            "inputs.face_restore_visibility": restore_visibility,
            "inputs.face_restore_model": restore_model,
        },
    })
    try:
        prompt_id = client.submit_workflow(wf)
        outputs = client.wait_for_completion(prompt_id, timeout=120)
        if len(outputs) != 1:
            raise ComfyUIError(f"Expected one image, got {len(outputs)}")
        ref = outputs[0]
        image_bytes = client.download_image(ref["filename"], ref.get("subfolder", ""), ref.get("type", "output"))
    except ComfyUIError as exc:
        raise click.ClickException(str(exc)) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(image_bytes)
    click.echo(f"Saved: {output}")


if __name__ == "__main__":
    main()
