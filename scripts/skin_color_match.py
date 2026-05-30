"""
Post-process body skin tone lock for Ananya carousel pipeline.

Runs after ReActor faceswap, before 1080x1920 resize.
Shifts skin pixels (face + arms, legs, neck-below-blend, décolletage) in LAB space
to match face_ref skin tone. As of 2026-05-29 the LAB shift is applied to BOTH
face and body skin (face-inclusion) so warm-cast scenes do not leave the
rendered face darker than the face_ref baseline via ReActor blend bleed-through.

Usage:
    match_body_skin_to_face_ref(slide_path, face_ref_path, out_path)

Standalone smoke test:
    python scripts/skin_color_match.py path/to/slide.png [--face-ref path/to/face_ref.png]
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml

DEFAULT_FACE_REF = Path("character/ananya/seeds_v2/face_ref_v2.png")
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@lru_cache(maxsize=1)
def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _model_path(key: str) -> Path:
    cfg = _load_config()
    path = cfg.get("models", {}).get(key)
    if not path:
        raise RuntimeError(f"config.yaml missing models.{key}")
    return Path(path)

# HSV skin range — tuned for South Asian M-tone (seed 334521876)
# Hue: 0-25 + 165-180 (warm skin tones, excludes pink clothing artefacts)
_SKIN_HSV_LOWER1 = np.array([0, 30, 60], dtype=np.uint8)
_SKIN_HSV_UPPER1 = np.array([25, 200, 255], dtype=np.uint8)
_SKIN_HSV_LOWER2 = np.array([165, 30, 60], dtype=np.uint8)
_SKIN_HSV_UPPER2 = np.array([180, 200, 255], dtype=np.uint8)

# How far below the top of the face bbox to start the cheek sample ROI (fraction of bbox height)
_CHEEK_TOP_FRAC = 0.45
_CHEEK_BOT_FRAC = 0.75
_CHEEK_LEFT_FRAC = 0.25
_CHEEK_RIGHT_FRAC = 0.75


# Maximum LAB channel shift magnitude — clamp to avoid over-correction.
# Restored to 25.0 / 12.0 on 2026-05-29 after red lehenga festive: warm-cast
# scenes need a strong lift to bring rendered skin to face_ref_v2 fair tone.
# The 8.0 trial was too soft. Pairs with face-inclusion (apply lift to face
# too) so both face and body get unified to face_ref tone, neutralising the
# scene's ambient warm/red cast that was making faces read as "darker".
_MAX_L_SHIFT = 25.0
_MAX_AB_SHIFT = 12.0

# Minimum body skin pixel count to attempt correction (skip if too few exposed pixels)
_MIN_SKIN_PIXELS = 200

# Gaussian sigma (px) for soft-feathering the skin mask before applying LAB delta.
# Hard binary mask + large delta L caused visible seam halos at body silhouette
# (orange tank carousel v3, 2026-05-25). Feather blends shift smoothly across edge.
_MASK_FEATHER_SIGMA = 8.0


@lru_cache(maxsize=1)
def _load_face_model():
    from ultralytics import YOLO
    return YOLO(str(_model_path("yolo_face_bbox")))


@lru_cache(maxsize=1)
def _load_seg_model():
    from ultralytics import YOLO
    return YOLO(str(_model_path("yolo_person_seg")))


@lru_cache(maxsize=8)
def _sample_face_skin_lab_cached(face_ref_str: str, mtime_ns: int) -> tuple[float, float, float]:
    """LRU-cached face_ref LAB sample keyed on (path, mtime). mtime_ns invalidates cache if face_ref changes."""
    return _sample_face_skin_lab(Path(face_ref_str))


def _bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def _lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2BGR)
    return np.clip(bgr * 255.0, 0, 255).astype(np.uint8)


def _sample_cheek_lab_from_bgr(
    bgr: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> tuple[float, float, float]:
    """Sample skin tone from lower-cheek ROI of an in-memory image given an optional face bbox.

    Shared core used by both the face_ref sampler and the in-slide sampler.
    """
    h, w = bgr.shape[:2]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        bh, bw = y2 - y1, x2 - x1
        ry1 = y1 + int(bh * _CHEEK_TOP_FRAC)
        ry2 = y1 + int(bh * _CHEEK_BOT_FRAC)
        rx1 = x1 + int(bw * _CHEEK_LEFT_FRAC)
        rx2 = x1 + int(bw * _CHEEK_RIGHT_FRAC)
    else:
        ry1, ry2 = int(h * 0.35), int(h * 0.65)
        rx1, rx2 = int(w * 0.3), int(w * 0.7)

    roi_bgr = bgr[ry1:ry2, rx1:rx2]
    roi_lab = _bgr_to_lab(roi_bgr)

    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(roi_hsv, _SKIN_HSV_LOWER1, _SKIN_HSV_UPPER1)
    m2 = cv2.inRange(roi_hsv, _SKIN_HSV_LOWER2, _SKIN_HSV_UPPER2)
    skin_mask = cv2.bitwise_or(m1, m2).astype(bool)

    if skin_mask.sum() < 50:
        skin_mask = np.ones(roi_bgr.shape[:2], dtype=bool)

    L = float(roi_lab[:, :, 0][skin_mask].mean())
    a = float(roi_lab[:, :, 1][skin_mask].mean())
    b = float(roi_lab[:, :, 2][skin_mask].mean())
    return L, a, b


def _sample_face_skin_lab(face_ref_path: Path) -> tuple[float, float, float]:
    """Sample target skin tone from lower-cheek ROI of face_ref (avoids eyes/lips/hair)."""
    bgr = cv2.imread(str(face_ref_path))
    if bgr is None:
        raise FileNotFoundError(f"face_ref not found: {face_ref_path}")

    model = _load_face_model()
    results = model(bgr, verbose=False)
    bbox = None
    if results and results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
        x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
        bbox = (int(x1), int(y1), int(x2), int(y2))

    return _sample_cheek_lab_from_bgr(bgr, bbox)


def _detect_face_bbox(img_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Returns (x1, y1, x2, y2) of the largest face, or None if not found."""
    model = _load_face_model()
    results = model(img_bgr, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes.xyxy) == 0:
        return None
    # Pick highest-confidence detection
    best_idx = int(results[0].boxes.conf.argmax())
    x1, y1, x2, y2 = results[0].boxes.xyxy[best_idx].cpu().numpy().astype(int)
    h, w = img_bgr.shape[:2]
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _person_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Returns boolean mask of person pixels via yolov8n-seg. Falls back to full image if no person detected."""
    model = _load_seg_model()
    h, w = img_bgr.shape[:2]
    results = model(img_bgr, verbose=False)

    if (results and results[0].masks is not None
            and len(results[0].masks.data) > 0):
        # Find person class (class 0 in COCO)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        person_indices = np.where(classes == 0)[0]
        if len(person_indices) > 0:
            # Union of all person masks (handles partial occlusion)
            combined = np.zeros((h, w), dtype=np.uint8)
            for idx in person_indices:
                mask_tensor = results[0].masks.data[idx].cpu().numpy()
                mask_resized = cv2.resize(mask_tensor, (w, h), interpolation=cv2.INTER_NEAREST)
                combined = cv2.bitwise_or(combined, (mask_resized > 0.5).astype(np.uint8))
            return combined.astype(bool)

    # Fallback: treat whole image as person region
    return np.ones((h, w), dtype=bool)


def _hsv_skin_filter(img_bgr: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    """Restrict to person_mask pixels that also match HSV skin range."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, _SKIN_HSV_LOWER1, _SKIN_HSV_UPPER1)
    m2 = cv2.inRange(hsv, _SKIN_HSV_LOWER2, _SKIN_HSV_UPPER2)
    skin = cv2.bitwise_or(m1, m2).astype(bool)
    return skin & person_mask


# Note: _face_exclusion_mask + _FACE_EXCLUSION_DILATE were removed 2026-05-29
# when match_body_skin_to_face_ref switched to face+body inclusion (lift the
# whole subject to face_ref tone, not just body). If a future scene needs to
# preserve face pixel-identical and only correct body, restore the helper from
# Git history (commit before bbb8e8b).


def _apply_lab_delta(
    img_bgr: np.ndarray,
    body_skin_mask: np.ndarray,
    target_lab: tuple[float, float, float],
) -> np.ndarray:
    """Compute mean LAB of body_skin_mask pixels, compute delta to target, shift those pixels."""
    if body_skin_mask.sum() < _MIN_SKIN_PIXELS:
        return img_bgr  # nothing to correct

    lab = _bgr_to_lab(img_bgr)  # float32 LAB

    L_vals = lab[:, :, 0][body_skin_mask]
    a_vals = lab[:, :, 1][body_skin_mask]
    b_vals = lab[:, :, 2][body_skin_mask]

    src_L = float(L_vals.mean())
    src_a = float(a_vals.mean())
    src_b = float(b_vals.mean())

    dL = float(np.clip(target_lab[0] - src_L, -_MAX_L_SHIFT, _MAX_L_SHIFT))
    da = float(np.clip(target_lab[1] - src_a, -_MAX_AB_SHIFT, _MAX_AB_SHIFT))
    db = float(np.clip(target_lab[2] - src_b, -_MAX_AB_SHIFT, _MAX_AB_SHIFT))

    print(f"  [skin_color_match] src LAB ({src_L:.1f}, {src_a:.1f}, {src_b:.1f}) "
          f"→ target ({target_lab[0]:.1f}, {target_lab[1]:.1f}, {target_lab[2]:.1f}) "
          f"delta ({dL:+.1f}, {da:+.1f}, {db:+.1f}) "
          f"pixels={int(body_skin_mask.sum())}")

    # Soft-feather mask: hard bool edges produce visible seam at silhouette when delta L is large.
    # Gaussian-blur a float [0,1] mask, then alpha-blend the LAB shift.
    mask_f = body_skin_mask.astype(np.float32)
    mask_blur = cv2.GaussianBlur(
        mask_f, (0, 0),
        sigmaX=_MASK_FEATHER_SIGMA, sigmaY=_MASK_FEATHER_SIGMA,
    )
    alpha = mask_blur[..., None]  # (H, W, 1) broadcasts over LAB channels

    shift = np.array([dL, da, db], dtype=np.float32)
    result_lab = lab + alpha * shift
    result_lab[..., 0] = np.clip(result_lab[..., 0], 0.0, 100.0)
    result_lab[..., 1] = np.clip(result_lab[..., 1], -128.0, 127.0)
    result_lab[..., 2] = np.clip(result_lab[..., 2], -128.0, 127.0)

    return _lab_to_bgr(result_lab)


def match_body_skin_to_face_ref(
    slide_path: Path,
    face_ref_path: Path,
    out_path: Path,
) -> None:
    """
    Main pipeline: detect face → person mask → HSV skin filter → LAB shift face+body.

    Both face skin and body skin get the same uniform LAB shift toward face_ref_v2
    cheek tone (sampled from the static reference). This neutralises warm/cool
    ambient casts (festive red, golden hour, indoor incandescent) that would
    otherwise leave the rendered face darker than the body once body alone gets
    lifted. Facial structure (eyes, lips, nose) is preserved by the LAB delta
    being uniform and tone-only — only colour shifts, never geometry.

    Saves corrected image to out_path (may be same as slide_path for in-place).
    """
    slide_path = Path(slide_path)
    face_ref_path = Path(face_ref_path)
    out_path = Path(out_path)

    img_bgr = cv2.imread(str(slide_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Slide not found: {slide_path}")

    # Target LAB: always face_ref_v2 cheek (fair, neutral-lit reference). Body +
    # face both get pulled toward this. The in-slide sampling option (tried on
    # 2026-05-29 v9c) was wrong-direction: it locked body to the dark-rendered
    # face, but the user actually wants both to be at the fair face_ref tone.
    face_ref_resolved = face_ref_path.resolve()
    target_lab = _sample_face_skin_lab_cached(
        str(face_ref_resolved),
        face_ref_resolved.stat().st_mtime_ns,
    )

    face_bbox = _detect_face_bbox(img_bgr)
    pmask = _person_mask(img_bgr)
    skin_mask = _hsv_skin_filter(img_bgr, pmask)

    # Apply LAB delta to BOTH face and body skin (no face exclusion). Reason:
    # warm-cast scenes (festive red, golden hour, indoor incandescent) darken
    # the rendered face away from face_ref_v2 tone via ReActor blend; lifting
    # face along with body unifies the whole subject to the fair reference.
    # This is a uniform LAB shift, not pixel substitution — facial structure
    # (eyes, lips, nose) is preserved; only tone shifts.
    full_skin_mask = skin_mask  # includes face skin

    corrected = _apply_lab_delta(img_bgr, full_skin_mask, target_lab)
    cv2.imwrite(str(out_path), corrected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone skin tone match smoke test")
    parser.add_argument("slide", type=Path, help="Path to input slide image")
    parser.add_argument("--face-ref", type=Path, default=DEFAULT_FACE_REF,
                        help="Path to face_ref image (default: character/ananya/seeds_v2/face_ref_v2.png)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default: slide_skinfix.png alongside input)")
    args = parser.parse_args()

    out = args.out or args.slide.parent / (args.slide.stem + "_skinfix" + args.slide.suffix)
    print(f"Input:    {args.slide}")
    print(f"Face ref: {args.face_ref}")
    print(f"Output:   {out}")

    match_body_skin_to_face_ref(args.slide, args.face_ref, out)
    print("Done.")
