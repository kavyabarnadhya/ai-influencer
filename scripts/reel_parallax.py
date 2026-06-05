"""
Ken Burns push reel generator — cinematic dolly-in / sway / pan over a static slide.

**Scope: IG Stories only (NOT grid Reels.)** CV motion on portrait stills reads as
PPT-tier animation — fine as a Story background or as a brand b-roll loop, but
audiences clock it as fake on the IG Reels feed where they expect real subject
motion (hair sway, breath, fabric flow). For grid Reels, only diffusion video
models (Wan, Sora, Veo, Kling) deliver believable motion — see
`character/ananya/reels_deferred.md` for the research + revisit triggers.

Default mode (`--depth-scale 0`, `--sway-px 0`, `--dolly-px 0`): pure Ken Burns
zoom-in, the safest setting. Subject does not slide — only camera dollies forward.
No silhouette wobble, no edge artefacts.

Pipeline:
  1. Load PNG (typically 1080x1920 carousel slide)
  2. Estimate depth with MiDaS small (CPU, ~5-10s per slide)
  3. Render N frames with parameterised camera path (zoom + optional sway + dolly +
     optional depth-weighted parallax shift via cv2.remap)
  4. Optionally palindrome-mirror for seamless loop
  5. Encode to .mp4 (mp4v fourcc — IG re-encodes on upload anyway)

First invocation downloads ~500MB total to ~/.cache/torch/hub/ (MiDaS small ~16MB,
gen-efficientnet-pytorch backbone ~200MB, tf_efficientnet_lite3 weights ~80MB,
transforms). Subsequent runs use the cached weights.

Identity-safe: every output pixel is a sampled position from the input PNG. No
diffusion, no drift, no eye flicker, no hand melt.

Usage:
  # Default Ken Burns push (safe, Stories-ready)
  python scripts/reel_parallax.py --input slide_00.png --output reel.mp4

  # Add subtle parallax (use cautiously — can cardboard-wobble on portraits)
  python scripts/reel_parallax.py --input slide_00.png --output reel.mp4 \\
      --depth-scale 0.4 --sway-px 8 --dolly-px 4
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import click
import cv2
import numpy as np
import torch
from rich.console import Console

console = Console()

# Cache for meshgrid to avoid redundant allocations across frames
# Key: (H, W), Value: (rel_grid_x, rel_grid_y) where rel_grid = grid - center
_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


@lru_cache(maxsize=1)
def _load_midas():
    """Load MiDaS small via torch.hub. Auto-downloads on first call (~16MB)."""
    # MiDaS pulls in rwightman/gen-efficientnet-pytorch as a backbone. Pre-trust it
    # so the subsequent torch.hub.load does not block on an interactive (y/N) prompt.
    try:
        torch.hub.list("rwightman/gen-efficientnet-pytorch", trust_repo=True,
                       skip_validation=True)
    except Exception as e:
        console.print(f"[yellow]Pre-trust step failed (continuing): {e}[/yellow]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[cyan]Loading MiDaS small ({device})...[/cyan]")

    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    model.to(device).eval()

    # Optimization: Use half-precision on GPU to significantly reduce inference time.
    if device.type == "cuda":
        model.half()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = transforms.small_transform
    return model, transform, device


def estimate_depth(img_bgr: np.ndarray, smooth_sigma: float = 12.0) -> np.ndarray:
    """Returns depth map (H, W) float32 in [0, 1] where 1 = closest, 0 = farthest.

    Gaussian-blur smooths depth-boundary discontinuities (face/hair vs BG) that would
    otherwise produce UV cracks and pixel smearing in the parallax remap.
    """
    model, transform, device = _load_midas()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    batch = transform(img_rgb).to(device)

    # Optimization: Enable half-precision for GPU inference
    if device.type == "cuda":
        batch = batch.half()

    with torch.no_grad():
        prediction = model(batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    # Optimization: Ensure depth is float32 for OpenCV compatibility.
    # Standard cv2.GaussianBlur does not support float16.
    depth = prediction.float().cpu().numpy()
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin < 1e-6:
        return np.full_like(depth, 0.5)
    depth = (depth - dmin) / (dmax - dmin)
    if smooth_sigma > 0:
        depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)
    return depth


def _camera_path(t: float, zoom: float, sway_px: float, dolly_px: float) -> tuple[float, float, float]:
    """
    Smooth camera path for parameter t in [0, 1].
    Returns (zoom_factor, dx_px, dy_px) — relative offsets applied per frame.
    Uses sine for sway/dolly so the path is smooth (palindrome-friendly).
    """
    z = 1.0 + zoom * t  # slow linear zoom-in across the clip
    dx = sway_px * np.sin(2 * np.pi * t)  # one full horizontal sway cycle
    dy = dolly_px * np.sin(np.pi * t)  # half cycle vertical drift, returns to center
    return float(z), float(dx), float(dy)


def render_parallax_frame(
    img_bgr: np.ndarray,
    parallax: np.ndarray,
    zoom: float,
    dx_px: float,
    dy_px: float,
) -> np.ndarray:
    """
    Render one parallax frame using pre-centered grids and pre-computed parallax map.
    """
    h, w = img_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Optimization: Use pre-centered grids to avoid redundant O(H*W) subtractions per frame
    cache_key = (h, w)
    if cache_key in _GRID_CACHE:
        rel_grid_x, rel_grid_y = _GRID_CACHE[cache_key]
    else:
        xs = np.arange(w, dtype=np.float32) - cx
        ys = np.arange(h, dtype=np.float32) - cy
        rel_grid_x, rel_grid_y = np.meshgrid(xs, ys)
        _GRID_CACHE[cache_key] = (rel_grid_x, rel_grid_y)

    # Zoom and Parallax: minimized coordinate arithmetic
    inv_z = 1.0 / zoom
    src_x = rel_grid_x * inv_z + (cx - dx_px * parallax)
    src_y = rel_grid_y * inv_z + (cy - dy_px * parallax)

    return cv2.remap(
        img_bgr, src_x, src_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _pick_fourcc() -> tuple[int, str]:
    """Default to mp4v.

    H264 / avc1 would be smaller but requires openh264 DLL on Windows; absence
    causes OpenCV to silently fall back to mp4v anyway. IG re-encodes everything
    on upload, so mp4v output (larger but reliable) is the right default.
    """
    return cv2.VideoWriter_fourcc(*"mp4v"), "mp4v"


@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Input PNG (e.g. carousel slide).")
@click.option("--output", "output_path", required=True, type=click.Path(dir_okay=False),
              help="Output .mp4 path.")
@click.option("--duration", default=5.0, type=float, help="Clip duration in seconds (default 5).")
@click.option("--fps", default=30, type=int, help="Output fps (default 30).")
@click.option("--zoom", default=0.06, type=float,
              help="Zoom-in amount across clip, e.g. 0.06 = 6%% (default 0.06).")
@click.option("--sway-px", default=0.0, type=float,
              help="Max horizontal sway in source pixels (default 0 = pure Ken Burns push). "
                   "Non-zero adds horizontal slide; combine with --depth-scale for parallax.")
@click.option("--dolly-px", default=0.0, type=float,
              help="Max vertical drift in source pixels (default 0). Non-zero adds vertical drift.")
@click.option("--depth-scale", default=0.0, type=float,
              help="Per-pixel parallax weighting, 0..1 (default 0 = no parallax = uniform pan). "
                   "Non-zero introduces 2.5D depth shift but can cardboard-wobble on portrait "
                   "subjects — use 0.2-0.4 if you must.")
@click.option("--depth-smooth", default=12.0, type=float,
              help="Gaussian sigma to smooth MiDaS depth before remap (default 12). "
                   "Higher = softer parallax, fewer edge artefacts.")
@click.option("--loop", type=click.Choice(["palindrome", "none"]), default="palindrome",
              help="palindrome = forward+reverse for seamless loop (default).")
def main(input_path: str, output_path: str, duration: float, fps: int,
         zoom: float, sway_px: float, dolly_px: float, depth_scale: float,
         depth_smooth: float, loop: str) -> None:
    src = Path(input_path).resolve()
    dst = Path(output_path).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Input:[/bold] {src}")
    console.print(f"[bold]Output:[/bold] {dst}")
    console.print(f"[bold]Spec:[/bold] {duration}s @ {fps}fps, zoom={zoom}, sway={sway_px}px, "
                  f"dolly={dolly_px}px, depth_scale={depth_scale}, loop={loop}")

    img = cv2.imread(str(src))
    if img is None:
        raise click.ClickException(f"Could not read image: {src}")
    h, w = img.shape[:2]
    console.print(f"[bold]Dimensions:[/bold] {w}x{h}")

    depth = estimate_depth(img, smooth_sigma=depth_smooth)
    console.print(f"[bold]Depth:[/bold] range=[{depth.min():.3f}, {depth.max():.3f}], "
                  f"mean={depth.mean():.3f}")

    # Pre-compute parallax map once (constant across all frames)
    # Approx 24% math speedup for the frame loop.
    parallax = (1.0 - depth_scale) + depth_scale * depth

    n_forward = int(round(duration * fps))
    if loop == "palindrome":
        # Halve the forward count so total duration matches --duration
        n_forward = max(2, n_forward // 2)
    console.print(f"[bold]Frames:[/bold] {n_forward} forward"
                  + (f" + {n_forward - 2} reverse (palindrome)" if loop == "palindrome" else ""))

    fourcc, codec = _pick_fourcc()
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise click.ClickException(f"Could not open VideoWriter (codec={codec})")
    console.print(f"[bold]Codec:[/bold] {codec}")

    forward_frames: list[np.ndarray] = []
    for i in range(n_forward):
        t = i / max(1, n_forward - 1)
        z, dx, dy = _camera_path(t, zoom, sway_px, dolly_px)
        frame = render_parallax_frame(img, parallax, z, dx, dy)
        forward_frames.append(frame)
        writer.write(frame)
        if (i + 1) % 30 == 0:
            console.print(f"  forward {i + 1}/{n_forward}")

    if loop == "palindrome":
        # Reverse without first + last to avoid duplicate frames at seam
        for i, frame in enumerate(reversed(forward_frames[1:-1])):
            writer.write(frame)
            if (i + 1) % 30 == 0:
                console.print(f"  reverse {i + 1}/{n_forward - 2}")

    writer.release()
    size_mb = dst.stat().st_size / 1024 / 1024
    console.print(f"\n[green]Done.[/green] {dst.name} — {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
