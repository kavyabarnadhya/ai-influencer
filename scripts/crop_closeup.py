"""Crop a chest-up or waist-up region from a full-body slide and resize to 1080x1920 (9:16).

Outfit pixels stay identical to the source slide — no regeneration. Use this to derive
"closeup" carousel slides from the highest-quality standing slide.

Usage:
    python scripts/crop_closeup.py --input output/.../slide_00_cand_0.png --framing chest_up
    python scripts/crop_closeup.py --input <path> --framing waist_up --output <out.png>
    python scripts/crop_closeup.py --input <path> --crop 80,40,720,1180   # manual box

Framing presets assume source is a portrait full-body shot with head near top:
    chest_up  : top 45% of image, centered horizontally
    waist_up  : top 65% of image, centered horizontally
    face_only : top 30% of image, centered horizontally
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

TARGET_W, TARGET_H = 1080, 1920  # Instagram Reels 9:16

PRESETS = {
    "chest_up": 0.45,
    "waist_up": 0.65,
    "face_only": 0.30,
}


def crop_to_9x16(img: Image.Image, top_fraction: float) -> Image.Image:
    """Crop top N% of image then enforce 9:16 by horizontal centering."""
    w, h = img.size
    crop_h = int(h * top_fraction)
    target_aspect = TARGET_W / TARGET_H  # 0.5625
    crop_w = int(crop_h * target_aspect)
    if crop_w > w:
        crop_w = w
        crop_h = int(crop_w / target_aspect)
    left = (w - crop_w) // 2
    return img.crop((left, 0, left + crop_w, crop_h))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--framing", choices=list(PRESETS.keys()), default="chest_up")
    p.add_argument("--crop", help="manual crop box: left,top,right,bottom (overrides --framing)")
    p.add_argument("--output", type=Path, help="output path (default: <input>_<framing>.png)")
    args = p.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    img = Image.open(args.input).convert("RGB")
    if args.crop:
        box = tuple(int(x) for x in args.crop.split(","))
        if len(box) != 4:
            print("ERROR: --crop needs 4 ints: left,top,right,bottom", file=sys.stderr)
            return 1
        cropped = img.crop(box)
    else:
        cropped = crop_to_9x16(img, PRESETS[args.framing])

    resized = cropped.resize((TARGET_W, TARGET_H), Image.LANCZOS)

    out = args.output or args.input.with_name(f"{args.input.stem}_{args.framing}.png")
    resized.save(out, "PNG")
    print(f"-> {out}  ({cropped.size[0]}x{cropped.size[1]} -> {TARGET_W}x{TARGET_H})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
