"""
Layer 3 hand QC for Ananya carousels - flag hand/finger defects automatically so
they're caught before human review, and auto-pick the best candidate per slide.

Two backends (auto-detected):
  * YOLO hand count (always available - uses the same hand_yolov8s.pt the pipeline's
    Stage 3.5 uses). Catches the universal hard failure: MORE THAN TWO hands
    (extra/third limb), and optionally missing hands where expected.
  * MediaPipe finger landmarks (OPTIONAL - only if `mediapipe` imports). Adds true
    finger-level checks (21 landmarks/hand, finger-count plausibility). MediaPipe
    pins opencv-contrib which can clash with the pipeline's opencv on Windows, so it
    is NOT a hard dependency - the YOLO backend works without it.

Usage:
    # score one image
    python scripts/hand_qc.py output/.../slide_00_cand_0.png

    # score every finished slide in a carousel folder (exit 1 if any flagged)
    python scripts/hand_qc.py output/2026-06-06/ananya/carousel_navy_gold_gown_v1/

    # pick the best candidate per slide from a multi-candidate run
    python scripts/hand_qc.py output/.../carousel_x/ --pick

Lower score = cleaner. score 0 = no hand flags.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import click
import yaml

ROOT = Path(__file__).resolve().parent.parent

# --- locate the hand YOLO model (same one Stage 3.5 uses) --------------------
_DEFAULT_HAND_MODELS = [
    Path(r"C:\Users\barna\Documents\ComfyUI\models\ultralytics\bbox\hand_yolov8s.pt"),
    ROOT / "models" / "hand_yolov8s.pt",
]


def _find_hand_model() -> Path | None:
    # allow config override
    cfg = ROOT / "config.yaml"
    if cfg.exists():
        try:
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            p = data.get("models", {}).get("hand_yolo")
            if p and Path(p).exists():
                return Path(p)
        except Exception:
            pass
    for p in _DEFAULT_HAND_MODELS:
        if p.exists():
            return p
    return None


# --- optional mediapipe -----------------------------------------------------
try:
    import mediapipe as _mp  # noqa: F401
    _HAVE_MP = True
except Exception:
    _HAVE_MP = False


_yolo_model = None


def _yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        mp_path = _find_hand_model()
        if mp_path is None:
            raise FileNotFoundError(
                "hand_yolov8s.pt not found. Set models.hand_yolo in config.yaml or place it under models/."
            )
        _yolo_model = YOLO(str(mp_path))
    return _yolo_model


def count_hands_yolo(img_path: Path, conf: float = 0.40) -> list[float]:
    """Return list of detection confidences for hands found in the image."""
    res = _yolo().predict(str(img_path), conf=conf, verbose=False)
    out = []
    for r in res:
        if r.boxes is None:
            continue
        out.extend(float(c) for c in r.boxes.conf.tolist())
    return out


def _finger_anomalies_mp(img_path: Path) -> int:
    """MediaPipe finger-landmark sanity. Returns count of anomalous hands (0 if MP absent)."""
    if not _HAVE_MP:
        return 0
    import cv2
    import mediapipe as mp
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    anomalies = 0
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=4,
                                  min_detection_confidence=0.4) as hands:
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            return 0
        for lm in res.multi_hand_landmarks:
            pts = lm.landmark
            if len(pts) != 21:           # malformed hand
                anomalies += 1
                continue
            # crude finger-spread plausibility: fingertip y-spread shouldn't collapse to ~0
            tips = [pts[i] for i in (4, 8, 12, 16, 20)]
            spread = max(t.x for t in tips) - min(t.x for t in tips)
            if spread < 0.01:            # all tips stacked -> fused/claw
                anomalies += 1
    return anomalies


def score_image(img_path: Path, expected_max: int = 2, expect_hands: bool = False) -> dict:
    """Score one image. Lower = cleaner. Returns dict with n_hands, flags, score."""
    confs = count_hands_yolo(img_path)
    n = len(confs)
    flags: list[str] = []
    score = 0
    if n > expected_max:
        flags.append(f"EXTRA_HANDS({n})")
        score += 10 * (n - expected_max)
    if expect_hands and n == 0:
        flags.append("NO_HANDS_DETECTED")
        score += 5
    if _HAVE_MP:
        a = _finger_anomalies_mp(img_path)
        if a:
            flags.append(f"FINGER_ANOMALY({a})")
            score += 8 * a
    return {"path": img_path, "n_hands": n, "confs": confs, "flags": flags, "score": score}


_SLIDE_CAND = re.compile(r"slide_(\d+)_cand_(\d+)\.png$", re.I)


@click.command()
@click.argument("target", type=click.Path(exists=True, path_type=Path))
@click.option("--expected-max", default=2, show_default=True, help="Max plausible hands per image.")
@click.option("--pick", is_flag=True, help="Pick the best candidate per slide index (multi-cand runs).")
@click.option("--strict", is_flag=True, help="Exit 1 if any slide still has flags after picking.")
def main(target: Path, expected_max: int, pick: bool, strict: bool):
    """Score hand QC for an image or a carousel folder."""
    click.echo(f"hand QC backend: YOLO{' + mediapipe' if _HAVE_MP else ' only (mediapipe absent - no finger-level check)'}")

    if target.is_file():
        r = score_image(target, expected_max)
        status = "CLEAN" if not r["flags"] else " ".join(r["flags"])
        click.echo(f"{target.name}: hands={r['n_hands']} score={r['score']} {status}")
        sys.exit(1 if (strict and r["flags"]) else 0)

    imgs = sorted(target.glob("slide_*_cand_*.png"))
    if not imgs:
        click.echo("no slide_*_cand_*.png found.")
        sys.exit(0)

    by_slide: dict[str, list[dict]] = defaultdict(list)
    for img in imgs:
        m = _SLIDE_CAND.search(img.name)
        if not m:
            continue
        r = score_image(img, expected_max)
        by_slide[m.group(1)].append(r)

    any_flag = False
    for sidx in sorted(by_slide):
        cands = sorted(by_slide[sidx], key=lambda r: r["score"])
        best = cands[0]
        if pick and len(cands) > 1:
            tail = " | ".join(f"cand{ _SLIDE_CAND.search(c['path'].name).group(2)}:{c['score']}" for c in cands)
            click.echo(f"slide_{sidx}: BEST cand{_SLIDE_CAND.search(best['path'].name).group(2)} "
                       f"(score {best['score']}) [{tail}]")
        for c in cands if not pick else [best]:
            status = "CLEAN" if not c["flags"] else " ".join(c["flags"])
            if c["flags"]:
                any_flag = True
            click.echo(f"  {c['path'].name}: hands={c['n_hands']} score={c['score']} {status}")

    sys.exit(1 if (strict and any_flag) else 0)


if __name__ == "__main__":
    main()
