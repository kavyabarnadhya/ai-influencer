"""Frame extractor for visual review of reel_parallax.py output (or any mp4).

Usage:
  python scripts/debug_extract_frames.py path/to/clip.mp4 [--count N]

Writes N downscaled sample frames evenly spaced across the clip duration to
./_frames_<stem>/ alongside the source mp4. Default N=6 covers first / quarter /
mid-forward / palindrome-seam / three-quarter / last. Use the Read tool on the
PNGs to inspect — useful because Claude Code's Read tool cannot watch mp4 directly.
"""
from __future__ import annotations

from pathlib import Path

import click
import cv2


@click.command()
@click.argument("src", type=click.Path(exists=True, dir_okay=False))
@click.option("--count", default=6, type=int, show_default=True,
              help="Number of sample frames to extract (evenly spaced).")
@click.option("--width", default=540, type=int, show_default=True,
              help="Downscale width for output PNGs (preserves aspect).")
def main(src: str, count: int, width: int) -> None:
    src_path = Path(src)
    out_dir = src_path.parent / f"_frames_{src_path.stem}"
    out_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(src_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {total} @ {fps} fps")
    if total < 1:
        raise click.ClickException(f"No frames in {src_path}")

    # Evenly spaced indices in [0, total-1], inclusive of endpoints.
    if count == 1:
        indices = [0]
    else:
        indices = [round(i * (total - 1) / (count - 1)) for i in range(count)]

    for slot, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"FAIL frame {idx}")
            continue
        h, w = frame.shape[:2]
        new_h = int(h * width / w)
        small = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)
        out = out_dir / f"{slot:02d}_idx{idx}.png"
        cv2.imwrite(str(out), small)
        print(f"Wrote {out.name}")

    cap.release()


if __name__ == "__main__":
    main()
