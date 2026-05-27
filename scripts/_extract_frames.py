"""Frame extractor for visual review of reel_parallax.py output (or any mp4).

Usage:
  python scripts/_extract_frames.py path/to/clip.mp4

Writes 6 downscaled sample frames (first, quarter, mid-forward, palindrome seam,
three-quarter, last) to ./_frames_<stem>/. Use Read tool on PNGs to inspect.
Useful because Claude Code's Read tool cannot watch mp4 directly.
"""
import sys
from pathlib import Path
import cv2

src = Path(sys.argv[1])
out_dir = src.parent / f"_frames_{src.stem}"
out_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture(str(src))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Total frames: {total} @ {fps} fps")

# Sample: first, 25%, mid, 75%, last forward, last
samples = {
    "00_first": 0,
    "01_quarter": total // 4,
    "02_mid_forward": total // 2 - 5,
    "03_palindrome_seam": total // 2,
    "04_three_quarter": (3 * total) // 4,
    "05_last": total - 1,
}

for name, idx in samples.items():
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        print(f"FAIL frame {idx}")
        continue
    # Downscale to save read budget
    h, w = frame.shape[:2]
    new_w = 540
    new_h = int(h * new_w / w)
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = out_dir / f"{name}_idx{idx}.png"
    cv2.imwrite(str(out), small)
    print(f"Wrote {out.name}")

cap.release()
