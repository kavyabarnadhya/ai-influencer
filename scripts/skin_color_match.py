"""
Post-process body skin tone lock for Ananya carousel pipeline.

Runs after ReActor faceswap, before 1080x1920 resize.
Shifts body skin pixels (arms, legs, neck-below-blend, décolletage) in LAB space
to match face_ref skin tone. Face region (from YOLO face bbox) is untouched so
ReActor output is fully preserved.

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

# Dilation radius for face mask exclusion zone (pixels at native res ~750px wide)
_FACE_EXCLUSION_DILATE = 10

# Maximum LAB channel shift magnitude — clamp to avoid over-correction
_MAX_L_SHIFT = 30.0
_MAX_AB_SHIFT = 15.0

# Minimum body skin pixel count to attempt correction (skip if too few exposed pixels)
_MIN_SKIN_PIXELS = 200


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


def _sample_face_skin_lab(face_ref_path: Path) -> tuple[float, float, float]:
    """Sample target skin tone from lower-cheek ROI of face_ref (avoids eyes/lips/hair)."""
    bgr = cv2.imread(str(face_ref_path))
    if bgr is None:
        raise FileNotFoundError(f"face_ref not found: {face_ref_path}")
    h, w = bgr.shape[:2]

    # Detect face bbox in the reference image
    model = _load_face_model()
    results = model(bgr, verbose=False)
    if results and results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
        x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
        # Clamp to image bounds
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        bh, bw = y2 - y1, x2 - x1
        # Cheek ROI: central horizontal strip, lower half of face
        ry1 = y1 + int(bh * _CHEEK_TOP_FRAC)
        ry2 = y1 + int(bh * _CHEEK_BOT_FRAC)
        rx1 = x1 + int(bw * _CHEEK_LEFT_FRAC)
        rx2 = x1 + int(bw * _CHEEK_RIGHT_FRAC)
    else:
        # Fallback: centre crop of the image (assumes face is centred in face_ref)
        ry1, ry2 = int(h * 0.35), int(h * 0.65)
        rx1, rx2 = int(w * 0.3), int(w * 0.7)

    roi_bgr = bgr[ry1:ry2, rx1:rx2]

    # Filter to skin pixels in the ROI
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(roi_hsv, _SKIN_HSV_LOWER1, _SKIN_HSV_UPPER1)
    m2 = cv2.inRange(roi_hsv, _SKIN_HSV_LOWER2, _SKIN_HSV_UPPER2)
    skin_mask = cv2.bitwise_or(m1, m2).astype(bool)

    if skin_mask.sum() < 50:
        # Not enough skin pixels — use entire ROI mean
        skin_mask = np.ones(roi_bgr.shape[:2], dtype=bool)

    # Optimization: Only convert skin pixels to LAB (much faster than full ROI conversion)
    skin_pixels_bgr = roi_bgr[skin_mask].reshape(1, -1, 3)
    skin_pixels_lab = _bgr_to_lab(skin_pixels_bgr).reshape(-1, 3)

    return tuple(skin_pixels_lab.mean(axis=0))


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
    """
    Restrict to person_mask pixels that also match HSV skin range.
    Optimization: Perform HSV conversion only on the bounding box of person_mask
    to avoid O(TotalPixels) overhead.
    """
    # Get bounding box of person_mask
    rows = np.any(person_mask, axis=1)
    cols = np.any(person_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return np.zeros_like(person_mask)

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    ymax += 1
    xmax += 1

    roi_bgr = img_bgr[ymin:ymax, xmin:xmax]
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(roi_hsv, _SKIN_HSV_LOWER1, _SKIN_HSV_UPPER1)
    m2 = cv2.inRange(roi_hsv, _SKIN_HSV_LOWER2, _SKIN_HSV_UPPER2)
    roi_skin = cv2.bitwise_or(m1, m2).astype(bool)

    full_skin = np.zeros(person_mask.shape, dtype=bool)
    full_skin[ymin:ymax, xmin:xmax] = roi_skin
    return full_skin & person_mask


def _face_exclusion_mask(img_bgr: np.ndarray, face_bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    """
    Returns boolean mask of pixels to EXCLUDE from correction (face region + dilation).
    Optimization: Perform dilation only on a small ROI surrounding the face
    bounding box to avoid O(TotalPixels) morphological overhead.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if face_bbox is None:
        return mask.astype(bool)

    x1, y1, x2, y2 = face_bbox
    if _FACE_EXCLUSION_DILATE > 0:
        d = _FACE_EXCLUSION_DILATE
        # Define ROI including the dilation padding
        ex1, ey1 = max(0, x1 - d), max(0, y1 - d)
        ex2, ey2 = min(w, x2 + d), min(h, y2 + d)

        roi_h, roi_w = ey2 - ey1, ex2 - ex1
        roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)

        # Map original face_bbox to ROI coordinates
        rx1, ry1 = x1 - ex1, y1 - ey1
        rx2, ry2 = x2 - ex1, y2 - ey1
        roi_mask[ry1:ry2, rx1:rx2] = 255

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * d + 1, 2 * d + 1)
        )
        roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)
        mask[ey1:ey2, ex1:ex2] = roi_mask
    else:
        mask[y1:y2, x1:x2] = 255

    return mask.astype(bool)


def _apply_lab_delta(
    img_bgr: np.ndarray,
    body_skin_mask: np.ndarray,
    target_lab: tuple[float, float, float],
) -> np.ndarray:
    """Compute mean LAB of body_skin_mask pixels, compute delta to target, shift those pixels."""
    num_pixels = int(body_skin_mask.sum())
    if num_pixels < _MIN_SKIN_PIXELS:
        return img_bgr  # nothing to correct

    # Optimization: Extract masked pixels ONLY and perform color conversion on them.
    # This avoids two full-image O(N) color conversions (BGR->LAB and LAB->BGR),
    # providing a ~5x speedup for typical carousel slides where skin is ~20% of pixels.
    skin_pixels_bgr = img_bgr[body_skin_mask].reshape(1, -1, 3)
    skin_pixels_lab = _bgr_to_lab(skin_pixels_bgr).reshape(-1, 3)
    src_means = skin_pixels_lab.mean(axis=0)

    deltas = np.clip(
        np.array(target_lab) - src_means,
        [-_MAX_L_SHIFT, -_MAX_AB_SHIFT, -_MAX_AB_SHIFT],
        [_MAX_L_SHIFT, _MAX_AB_SHIFT, _MAX_AB_SHIFT]
    )
    dL, da, db = deltas

    print(f"  [skin_color_match] src LAB ({src_means[0]:.1f}, {src_means[1]:.1f}, {src_means[2]:.1f}) "
          f"→ target ({target_lab[0]:.1f}, {target_lab[1]:.1f}, {target_lab[2]:.1f}) "
          f"delta ({dL:+.1f}, {da:+.1f}, {db:+.1f}) "
          f"pixels={num_pixels}")

    # Apply shift in LAB space and clip to valid ranges
    skin_pixels_lab += deltas
    np.clip(
        skin_pixels_lab,
        [0, -128, -128],
        [100, 127, 127],
        out=skin_pixels_lab
    )

    # Convert corrected skin pixels back to BGR
    res_skin_bgr = _lab_to_bgr(skin_pixels_lab.reshape(1, -1, 3)).reshape(-1, 3)

    # Write corrected pixels back into the original image
    result = img_bgr.copy()
    result[body_skin_mask] = res_skin_bgr
    return result


def match_body_skin_to_face_ref(
    slide_path: Path,
    face_ref_path: Path,
    out_path: Path,
) -> None:
    """
    Main pipeline: detect face → person mask → HSV skin filter → exclude face → LAB shift body.
    Face region is pixel-identical to input (ReActor output preserved).
    Saves corrected image to out_path (may be same as slide_path for in-place).
    """
    slide_path = Path(slide_path)
    face_ref_path = Path(face_ref_path)
    out_path = Path(out_path)

    img_bgr = cv2.imread(str(slide_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Slide not found: {slide_path}")

    # Cache target LAB by (path, mtime) — avoids re-sampling face_ref on every slide
    face_ref_resolved = face_ref_path.resolve()
    target_lab = _sample_face_skin_lab_cached(
        str(face_ref_resolved),
        face_ref_resolved.stat().st_mtime_ns,
    )

    face_bbox = _detect_face_bbox(img_bgr)
    pmask = _person_mask(img_bgr)
    skin_mask = _hsv_skin_filter(img_bgr, pmask)
    face_excl = _face_exclusion_mask(img_bgr, face_bbox)

    body_skin_mask = skin_mask & ~face_excl

    corrected = _apply_lab_delta(img_bgr, body_skin_mask, target_lab)
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
