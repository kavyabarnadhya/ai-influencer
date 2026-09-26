"""Assemble a carousel's still images into a vertical Ken Burns slideshow reel.

Pure ffmpeg pipeline (no moviepy, no new pip deps) — CV-only, no video model.
For a real-motion (hair/fabric/blink) reel, an actual img2vid model is required;
see character/ananya/reels_deferred.md. This tool produces the pan/zoom
slideshow style, explicitly NOT a substitute for that when real motion was asked
for elsewhere — it's its own deliverable: a carousel-to-reel repost format.

Pipeline:
  1. Discover slide_*.png in the input directory (sorted).
  2. Per slide: Ken Burns zoom clip (~duration-per-slide seconds, alternating
     zoom-in/zoom-out for rhythm) at 1080x1920, via ffmpeg zoompan.
  3. Crossfade-concat all clips with ffmpeg's xfade filter.
  4. Overlay a bold text hook on the first N seconds, pulled from the first
     non-empty line of the carousel's caption.txt (or --hook to override).
  5. If a *.mp3 exists in the input directory (or --music points at one), mux
     it in, trimmed/looped to the video's duration. Otherwise silent — drop an
     mp3 into the carousel folder later and rerun with --music.
  6. Encode h264 / yuv420p, sized to stay comfortably under 30MB for a
     ~15-20s clip at 1080x1920.

Usage:
    python scripts/reel_from_carousel.py --carousel-dir output/2026-09-26/ananya/restart/carousel_one_dress_three_ways
    python scripts/reel_from_carousel.py --carousel-dir <dir> --music <dir>/trending_audio.mp3
    python scripts/reel_from_carousel.py --carousel-dir <dir> --hook "one dress, zero decisions" --seconds-per-slide 2.5
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import click
from rich.console import Console

console = Console(safe_box=True, legacy_windows=False)

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def _strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text).strip()


def _wrap_hook(text: str, max_chars_per_line: int = 24, max_lines: int = 3) -> str:
    """Greedy word-wrap for the drawtext overlay; ffmpeg drawtext renders \\n as a line break."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars_per_line and current:
            lines.append(current)
            current = word
        else:
            current = candidate
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    return "\n".join(lines)
ROOT = Path(__file__).resolve().parent.parent

TARGET_W, TARGET_H = 1080, 1920
FPS = 30
CROSSFADE_S = 0.5
HOOK_VISIBLE_S = 2.5


def _run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
                           capture_output=True, text=True)
    if proc.returncode != 0:
        raise click.ClickException(f"ffmpeg failed:\n{proc.stderr[-4000:]}")


def _find_slides(carousel_dir: Path) -> list[Path]:
    slides = sorted(carousel_dir.glob("slide_*.png"))
    if not slides:
        raise click.ClickException(f"No slide_*.png found in {carousel_dir}")
    return slides


def _read_hook(carousel_dir: Path, override: str | None) -> str:
    if override:
        return _strip_emoji(override)
    caption_path = carousel_dir / "caption.txt"
    if not caption_path.exists():
        return ""
    for line in caption_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return _strip_emoji(line)
    return ""


def _escape_drawtext(text: str) -> str:
    # ffmpeg filtergraph syntax treats \ : ' , [ ] as special even inside a
    # quoted value passed via -vf; escape each for the drawtext text= literal.
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\u2019")
    text = text.replace(",", "\\,")
    text = text.replace("[", "\\[").replace("]", "\\]")
    return text


def _find_music(carousel_dir: Path, override: str | None) -> Path | None:
    if override:
        p = Path(override)
        return p if p.exists() else None
    mp3s = sorted(carousel_dir.glob("*.mp3"))
    return mp3s[0] if mp3s else None


def _build_slide_clip(src: Path, dst: Path, duration: float, zoom_in: bool) -> None:
    """One Ken Burns clip: zoom in on odd slides, zoom out on even, for rhythm."""
    frames = max(2, round(duration * FPS))
    if zoom_in:
        z_expr = f"min(zoom+{0.4 / frames:.6f},1.08)"
        start_zoom = "1.0"
    else:
        # zoompan has no native start-zoomed-out-then-in-reverse primitive without
        # keeping state, so approximate zoom-out via a decreasing expression seeded high.
        z_expr = f"if(eq(on,0),1.08,max(zoom-{0.4 / frames:.6f},1.0))"
        start_zoom = "1.08"
    vf = (
        f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_W}:{TARGET_H},"
        f"zoompan=z='{z_expr}':d={frames}:s={TARGET_W}x{TARGET_H}:fps={FPS}"
    )
    _run_ffmpeg([
        "-loop", "1", "-i", str(src),
        "-vf", vf,
        "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(dst),
    ])


def _crossfade_concat(clips: list[Path], durations: list[float], dst: Path) -> None:
    """Chain xfade across N clips. Each xfade offset = cumulative duration minus overlaps so far."""
    if len(clips) == 1:
        shutil.copy(clips[0], dst)
        return

    inputs: list[str] = []
    for c in clips:
        inputs += ["-i", str(c)]

    filter_parts = []
    prev_label = "0:v"
    cumulative = durations[0]
    for i in range(1, len(clips)):
        offset = cumulative - CROSSFADE_S
        out_label = f"v{i}" if i < len(clips) - 1 else "vout"
        filter_parts.append(
            f"[{prev_label}][{i}:v]xfade=transition=fade:duration={CROSSFADE_S}:offset={offset:.3f}[{out_label}]"
        )
        prev_label = out_label
        cumulative += durations[i] - CROSSFADE_S

    filtergraph = ";".join(filter_parts)
    _run_ffmpeg([
        *inputs,
        "-filter_complex", filtergraph,
        "-map", "[vout]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(dst),
    ])


def _overlay_hook(src: Path, dst: Path, hook: str, font: str) -> None:
    if not hook:
        shutil.copy(src, dst)
        return
    text = _escape_drawtext(_wrap_hook(hook))
    font_escaped = font.replace("\\", "/").replace(":", "\\:")
    drawtext = (
        f"drawtext=fontfile='{font_escaped}':text='{text}':fontcolor=white:fontsize=64:"
        f"borderw=4:bordercolor=black:x=(w-text_w)/2:y=h*0.12:"
        f"enable='between(t,0,{HOOK_VISIBLE_S})':line_spacing=8"
    )
    _run_ffmpeg([
        "-i", str(src),
        "-vf", drawtext,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(dst),
    ])


def _mux_audio(video_src: Path, music: Path | None, dst: Path, duration: float) -> None:
    if music is None:
        shutil.copy(video_src, dst)
        return
    _run_ffmpeg([
        "-i", str(video_src),
        "-stream_loop", "-1", "-i", str(music),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-t", str(duration),
        "-shortest",
        str(dst),
    ])


@click.command()
@click.option("--carousel-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Carousel output directory containing slide_*.png and (optionally) caption.txt.")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Output .mp4 path. Defaults to <carousel-dir>/reel.mp4.")
@click.option("--seconds-per-slide", type=float, default=3.0, show_default=True)
@click.option("--hook", default=None, help="Override hook text (default: first line of caption.txt).")
@click.option("--music", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Path to an mp3. Defaults to the first *.mp3 found in --carousel-dir, if any.")
@click.option("--font", default=r"C:\Windows\Fonts\impact.ttf", show_default=True,
              help="Font file for the hook text overlay.")
def main(carousel_dir: Path, output: Path | None, seconds_per_slide: float,
         hook: str | None, music: str | None, font: str) -> None:
    carousel_dir = carousel_dir.resolve()
    output = (output or carousel_dir / "reel.mp4").resolve()
    slides = _find_slides(carousel_dir)
    hook_text = _read_hook(carousel_dir, hook)
    music_path = _find_music(carousel_dir, music)

    console.print(f"[bold]Carousel:[/bold] {carousel_dir}")
    console.print(f"[bold]Slides:[/bold] {len(slides)} ({', '.join(s.name for s in slides)})")
    console.print(f"[bold]Hook:[/bold] {hook_text or '(none found)'}")
    console.print(f"[bold]Music:[/bold] {music_path or '(none — silent output)'}")

    tmp_dir = carousel_dir / "_reel_tmp"
    tmp_dir.mkdir(exist_ok=True)
    try:
        clip_paths = []
        durations = []
        for i, slide in enumerate(slides):
            clip_path = tmp_dir / f"clip_{i:02d}.mp4"
            console.print(f"[cyan]Rendering Ken Burns clip {i+1}/{len(slides)}...[/cyan]")
            _build_slide_clip(slide, clip_path, seconds_per_slide, zoom_in=(i % 2 == 0))
            clip_paths.append(clip_path)
            durations.append(seconds_per_slide)

        total_duration = sum(durations) - CROSSFADE_S * (len(clip_paths) - 1)

        console.print("[cyan]Crossfading slides together...[/cyan]")
        concat_path = tmp_dir / "concat.mp4"
        _crossfade_concat(clip_paths, durations, concat_path)

        console.print("[cyan]Overlaying hook text...[/cyan]")
        hooked_path = tmp_dir / "hooked.mp4"
        _overlay_hook(concat_path, hooked_path, hook_text, font)

        console.print("[cyan]Muxing audio...[/cyan]")
        _mux_audio(hooked_path, music_path, output, total_duration)

        size_mb = output.stat().st_size / 1024 / 1024
        console.print(f"\n[green]Done.[/green] {output} — {size_mb:.2f} MB, ~{total_duration:.1f}s")
        if size_mb >= 30:
            console.print("[yellow]Warning: output is at/above 30MB target.[/yellow]")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
