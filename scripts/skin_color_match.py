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
# History: 30/15 → 25/12 (2026-05-29, warm scenes needed strong lift) → 10/8 (2026-05-31).
# The 25/12 caps OVER-corrected warm-cast indoor scenes: face_ref_v2 cheek L≈61 vs
# warm-tungsten-rendered skin L≈49 = +12 L lift bleached skin to a plastic foundation
# look (red halter vanity carousel). L capped at 10 keeps a natural warm-lit tone.
# AB lowered 12→8 in step: the large a*/b* pull was scrubbing the scene's warm ambient
# cast off the skin (part of the same plastic look) — a gentler AB cap lets some warm
# ambience read through so skin sits in the scene instead of looking colour-corrected.
# Pairs with face-inclusion (lift applied to face + body together).
_MAX_L_SHIFT = 10.0
_MAX_AB_SHIFT = 8.0

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
    # Optimization: In-place multiplication on f32 cast saves an O(H*W) allocation.
    f32 = bgr.astype(np.float32)
    f32 *= np.float32(1.0 / 255.0)
    return cv2.cvtColor(f32, cv2.COLOR_BGR2Lab)


def _lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    # Optimization: lab is already float32 from _bgr_to_lab or shifts.
    bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    # Optimization: cv2.convertScaleAbs is significantly faster than NumPy scale/clip/cast.
    return cv2.convertScaleAbs(bgr, alpha=255.0)


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
    skin_mask = cv2.bitwise_or(m1, m2)

    # Optimization: cv2.countNonZero and cv2.mean are significantly faster than NumPy equivalents.
    if cv2.countNonZero(skin_mask) < 50:
        skin_mask = None

    L, a, b = cv2.mean(roi_lab, mask=skin_mask)[:3]
    return float(L), float(a), float(b)


def _sample_face_skin_lab(face_ref_path: Path) -> tuple[float, float, float]:
    """Sample target skin tone from lower-cheek ROI of face_ref (avoids eyes/lips/hair)."""
    bgr = cv2.imread(str(face_ref_path))
    if bgr is None:
        raise FileNotFoundError(f"face_ref not found: {face_ref_path}")

    bbox = _detect_face_bbox(bgr)
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
    """Returns uint8 mask [0, 255] of person pixels via yolov8n-seg. Falls back to full image if no person detected."""
    model = _load_seg_model()
    h, w = img_bgr.shape[:2]
    results = model(img_bgr, verbose=False)

    if (results and results[0].masks is not None
            and len(results[0].masks.data) > 0):
        # Find person class (class 0 in COCO)
        classes = results[0].boxes.cls
        person_indices = (classes == 0).nonzero(as_tuple=True)[0]
        if len(person_indices) > 0:
            # Optimization: Combine all person masks into a single union mask
            # on the GPU/torch side before a single resize call. This avoids
            # the loop of individual CPU resizes and bitwise_or operations.
            # Scaling to [0, 255] on the GPU before moving to CPU avoids a
            # full-resolution O(H*W) multiplication.
            combined_mask_u8 = (results[0].masks.data[person_indices].any(dim=0).byte() * 255).cpu().numpy()
            mask_resized = cv2.resize(combined_mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
            return mask_resized

    # Fallback: treat whole image as person region
    return np.full((h, w), 255, dtype=np.uint8)


def _hsv_skin_filter(img_bgr: np.ndarray, person_mask_u8: np.ndarray) -> np.ndarray:
    """Restrict to person_mask pixels that also match HSV skin range. Returns uint8 [0, 255]."""
    # Optimization: ROI-based processing. Approx 3x speedup for typical portraits.
    x, y, w_box, h_box = cv2.boundingRect(person_mask_u8)
    if w_box == 0 or h_box == 0:
        return np.zeros_like(person_mask_u8)

    rmin, rmax, cmin, cmax = y, y + h_box - 1, x, x + w_box - 1

    roi_bgr = img_bgr[rmin:rmax+1, cmin:cmax+1]
    roi_person_u8 = person_mask_u8[rmin:rmax+1, cmin:cmax+1]

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, _SKIN_HSV_LOWER1, _SKIN_HSV_UPPER1)
    m2 = cv2.inRange(hsv, _SKIN_HSV_LOWER2, _SKIN_HSV_UPPER2)
    skin_roi = cv2.bitwise_or(m1, m2)

    # Intersection of skin and person in ROI
    skin_roi = cv2.bitwise_and(skin_roi, roi_person_u8)

    full_skin_mask = np.zeros_like(person_mask_u8)
    full_skin_mask[rmin:rmax+1, cmin:cmax+1] = skin_roi
    return full_skin_mask


# Note: _face_exclusion_mask + _FACE_EXCLUSION_DILATE were removed 2026-05-29
# when match_body_skin_to_face_ref switched to face+body inclusion (lift the
# whole subject to face_ref tone, not just body). If a future scene needs to
# preserve face pixel-identical and only correct body, restore the helper from
# Git history (commit before bbb8e8b).


def _apply_lab_delta(
    img_bgr: np.ndarray,
    body_skin_mask_u8: np.ndarray,
    target_lab: tuple[float, float, float],
) -> np.ndarray:
    """Compute mean LAB of body_skin_mask_u8 pixels, compute delta to target, shift those pixels."""
    # Optimization: cv2.countNonZero is significantly faster than NumPy .sum() on boolean masks.
    n_pixels = cv2.countNonZero(body_skin_mask_u8)
    if n_pixels < _MIN_SKIN_PIXELS:
        return img_bgr  # nothing to correct

    # Optimization: Reuse the already calculated uint8 mask for bounding box discovery.
    x, y, w_box, h_box = cv2.boundingRect(body_skin_mask_u8)
    rmin, rmax, cmin, cmax = y, y + h_box - 1, x, x + w_box - 1

    # Add padding for Gaussian blur (3*sigma is safe)
    pad = int(_MASK_FEATHER_SIGMA * 3)
    h, w = img_bgr.shape[:2]
    rmin = max(0, rmin - pad)
    rmax = min(h - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(w - 1, cmax + pad)

    roi_bgr = img_bgr[rmin:rmax+1, cmin:cmax+1]
    roi_mask_u8 = body_skin_mask_u8[rmin:rmax+1, cmin:cmax+1]
    roi_lab = _bgr_to_lab(roi_bgr)

    # Optimization: cv2.mean is significantly faster than NumPy indexing.
    src_L, src_a, src_b = cv2.mean(roi_lab, mask=roi_mask_u8)[:3]

    dL = float(np.clip(target_lab[0] - src_L, -_MAX_L_SHIFT, _MAX_L_SHIFT))
    da = float(np.clip(target_lab[1] - src_a, -_MAX_AB_SHIFT, _MAX_AB_SHIFT))
    db = float(np.clip(target_lab[2] - src_b, -_MAX_AB_SHIFT, _MAX_AB_SHIFT))

    print(f"  [skin_color_match] src LAB ({src_L:.1f}, {src_a:.1f}, {src_b:.1f}) "
          f"→ target ({target_lab[0]:.1f}, {target_lab[1]:.1f}, {target_lab[2]:.1f}) "
          f"delta ({dL:+.1f}, {da:+.1f}, {db:+.1f}) "
          f"pixels={n_pixels}")

    # Optimization: Downsampled Gaussian blur for mask smoothing.
    # Smoothing large 1080p+ ROI masks with Gaussian blur is expensive.
    # Downsampling by 4x before blurring and upscaling back provides a
    # ~10x speedup with negligible impact on feathering quality.
    h_roi, w_roi = roi_mask_u8.shape[:2]
    mask_small = cv2.resize(roi_mask_u8, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    mask_small_blur = cv2.GaussianBlur(
        mask_small, (0, 0),
        sigmaX=_MASK_FEATHER_SIGMA / 4.0, sigmaY=_MASK_FEATHER_SIGMA / 4.0,
    )
    mask_blur_u8 = cv2.resize(mask_small_blur, (w_roi, h_roi), interpolation=cv2.INTER_LINEAR)

    # Optimization: In-place scaling saves an allocation.
    alpha_2d = mask_blur_u8.astype(np.float32)
    alpha_2d *= np.float32(1.0 / 255.0)

    # Optimization: Use per-channel in-place additions with 1D alpha.
    # This avoids the large (H, W, 3) float allocation for (alpha * shift).
    if dL != 0:
        roi_lab[..., 0] += alpha_2d * dL
    if da != 0:
        roi_lab[..., 1] += alpha_2d * da
    if db != 0:
        roi_lab[..., 2] += alpha_2d * db

    # In-place clipping
    np.clip(roi_lab[..., 0], 0.0, 100.0, out=roi_lab[..., 0])
    np.clip(roi_lab[..., 1], -128.0, 127.0, out=roi_lab[..., 1])
    np.clip(roi_lab[..., 2], -128.0, 127.0, out=roi_lab[..., 2])

    roi_result_bgr = _lab_to_bgr(roi_lab)

    # Optimization: ROI-based patching into a copy avoids full-frame intermediate allocations.
    img_result = img_bgr.copy()
    img_result[rmin:rmax+1, cmin:cmax+1] = roi_result_bgr
    return img_result


def match_body_skin_to_face_ref(
    slide_path: Path | None,
    face_ref_path: Path,
    out_path: Path | None,
    img_bgr: np.ndarray | None = None,
) -> np.ndarray:
    """
    Main pipeline: detect face → person mask → HSV skin filter → LAB shift face+body.

    Both face skin and body skin get the same uniform LAB shift toward face_ref_v2
    cheek tone (sampled from the static reference). This neutralises warm/cool
    ambient casts (festive red, golden hour, indoor incandescent) that would
    otherwise leave the rendered face darker than the body once body alone gets
    lifted. Facial structure (eyes, lips, nose) is preserved by the LAB delta
    being uniform and tone-only — only colour shifts, never geometry.

    If img_bgr is provided, slide_path is ignored (skips disk read).
    Saves corrected image to out_path (may be same as slide_path for in-place).
    If out_path is None, skips disk write.
    Returns the corrected image array (BGR) to allow callers to skip redundant I/O.
    """
    if slide_path is not None:
        slide_path = Path(slide_path)
    face_ref_path = Path(face_ref_path)
    if out_path is not None:
        out_path = Path(out_path)

    if img_bgr is None:
        if slide_path is None:
            raise ValueError("Either slide_path or img_bgr must be provided")
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
    # Optimization: Use compression level 3 for a balance of speed and file size.
    if out_path is not None:
        cv2.imwrite(str(out_path), corrected, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return corrected


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
