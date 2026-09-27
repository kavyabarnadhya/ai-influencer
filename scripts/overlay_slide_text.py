"""Burn on-image hook/recap text onto a single carousel still (PNG), face-safe.

Reuses reel_from_carousel.py's face-detection + auto-fit-font machinery
(same Haar cascade, same shrink-until-it-fits logic, same face-safe zone
placement) so a static image gets the identical guarantee already proven
for the reel pipeline: text never overlaps the detected face, never gets
clipped mid-sentence.

Renders with PIL directly (not ffmpeg drawtext) since this is a still, not
a video frame — simpler and avoids ffmpeg's filtergraph string-escaping.

Usage:
    python scripts/overlay_slide_text.py --input path/to/slide_00.png --output path/to/slide_00.png --text "one blue dress, three moods"
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from reel_from_carousel import (  # noqa: E402
    FACE_MARGIN, LINE_SPACING, MAX_FONT_SIZE, MIN_FONT_SIZE, TEXT_MAX_WIDTH,
    TEXT_SIDE_MARGIN, _detect_face_box, _face_safe_zone, _fit_hook,
)


@click.command()
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output", "output_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--text", required=True, help="Text to burn onto the image.")
@click.option("--font", default=r"C:\Windows\Fonts\impact.ttf", show_default=True)
def main(input_path: Path, output_path: Path, text: str, font: str):
    img = Image.open(input_path).convert("RGB")
    w, h = img.size

    face_box = _detect_face_box(input_path)
    zone_y, zone_h, zone_name = _face_safe_zone(face_box, w, h)
    zone_h = max(zone_h, MIN_FONT_SIZE)
    click.echo(f"Face box: {face_box or 'not detected'} -> zone: {zone_name} (y={zone_y}, h={zone_h})")

    wrapped, size = _fit_hook(text, font, TEXT_MAX_WIDTH, zone_h)
    click.echo(f"Fitted at fontsize={size}, {wrapped.count(chr(10)) + 1} line(s)")

    draw = ImageDraw.Draw(img)
    pil_font = ImageFont.truetype(font, size)
    lines = wrapped.split("\n")
    line_h = pil_font.getbbox("Ag")[3] - pil_font.getbbox("Ag")[1]
    block_h = line_h * len(lines) + LINE_SPACING * (len(lines) - 1)

    if zone_name in ("top", "top (above face)"):
        y = zone_y
    else:
        y = zone_y + (zone_h - block_h) // 2

    border_w = 4
    for line in lines:
        line_w = pil_font.getbbox(line)[2]
        x = (w - line_w) // 2
        # black outline for contrast, then white fill
        for dx in range(-border_w, border_w + 1):
            for dy in range(-border_w, border_w + 1):
                if dx * dx + dy * dy <= border_w * border_w:
                    draw.text((x + dx, y + dy), line, font=pil_font, fill="black")
        draw.text((x, y), line, font=pil_font, fill="white")
        y += line_h + LINE_SPACING

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    click.echo(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
