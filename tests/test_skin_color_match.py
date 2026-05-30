"""
Unit tests for the face-inclusion skin lock (scripts/skin_color_match.py).

Deterministic, CPU-only — no YOLO model load, no GPU. Covers the hot-path
LAB shift (_apply_lab_delta): min-pixel skip, shift-toward-target, clamp,
Gaussian feather; plus _hsv_skin_filter and the cheek sampler. Also guards
that _face_exclusion_mask stays removed (face-inclusion regression guard).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import skin_color_match as scm  # noqa: E402


def _lab_at(img_bgr, y, x):
    """Mean LAB of a small 5x5 window centred on (y, x)."""
    patch = img_bgr[y - 2:y + 3, x - 2:x + 3]
    lab = scm._bgr_to_lab(patch)
    return float(lab[..., 0].mean()), float(lab[..., 1].mean()), float(lab[..., 2].mean())


def _skin_bgr(hue=10, s=120, v=200):
    """A single BGR colour that falls inside the HSV skin range."""
    px = np.array([[[hue, s, v]]], dtype=np.uint8)
    return cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]


# --- _apply_lab_delta ---

def test_apply_lab_delta_skips_below_min_pixels():
    img = np.full((50, 50, 3), 150, np.uint8)
    mask = np.zeros((50, 50), bool)
    mask[:5, :5] = True  # 25 px < _MIN_SKIN_PIXELS (200)
    out = scm._apply_lab_delta(img, mask, (60.0, 10.0, 15.0))
    assert np.array_equal(out, img), "below min-pixel mask must return image unchanged"


def test_apply_lab_delta_shifts_toward_target():
    img = np.full((200, 200, 3), 0, np.uint8)
    img[50:150, 50:150] = _skin_bgr()           # 100x100 skin patch
    mask = np.zeros((200, 200), bool)
    mask[50:150, 50:150] = True

    src_L = _lab_at(img, 100, 100)[0]
    target = (src_L + 8.0, 5.0, 10.0)            # +8 L is within _MAX_L_SHIFT (25)
    out = scm._apply_lab_delta(img, mask, target)

    out_L = _lab_at(out, 100, 100)[0]            # centre = full alpha
    assert out_L > src_L + 4.0, "centre L should move up toward target"
    assert abs(out_L - target[0]) < 4.0, "centre L should land near (unclamped) target"


def test_apply_lab_delta_clamps_large_delta():
    img = np.full((200, 200, 3), 0, np.uint8)
    img[50:150, 50:150] = _skin_bgr()
    mask = np.zeros((200, 200), bool)
    mask[50:150, 50:150] = True

    src_L = _lab_at(img, 100, 100)[0]
    out = scm._apply_lab_delta(img, mask, (src_L + 100.0, 0.0, 0.0))  # absurd target
    out_L = _lab_at(out, 100, 100)[0]
    achieved = out_L - src_L
    assert achieved <= scm._MAX_L_SHIFT + 2.0, "L shift must be clamped to _MAX_L_SHIFT"
    assert achieved > scm._MAX_L_SHIFT - 4.0, "clamped shift should reach near the cap"


def test_apply_lab_delta_feathers_edge():
    """Feather: a pixel just outside the mask gets a partial shift; a far pixel none."""
    img = np.full((200, 200, 3), 0, np.uint8)
    img[50:150, 50:150] = _skin_bgr()
    mask = np.zeros((200, 200), bool)
    mask[50:150, 50:150] = True

    out = scm._apply_lab_delta(img, mask, (_lab_at(img, 100, 100)[0] + 20.0, 0.0, 0.0))

    # Just outside the mask edge (within ~σ): should change a little.
    near_before = _lab_at(img, 154, 100)[0]
    near_after = _lab_at(out, 154, 100)[0]
    assert near_after - near_before > 0.5, "edge feather should bleed shift just outside mask"

    # Far from the mask: untouched.
    far_before = _lab_at(img, 195, 195)[0]
    far_after = _lab_at(out, 195, 195)[0]
    assert abs(far_after - far_before) < 0.5, "pixels far from mask must be unchanged"


# --- _hsv_skin_filter ---

def test_hsv_skin_filter_keeps_skin_drops_nonskin_and_offmask():
    img = np.zeros((20, 30, 3), np.uint8)
    img[:, :10] = _skin_bgr()                    # skin, inside person mask
    img[:, 10:20] = _skin_bgr()                  # skin, OUTSIDE person mask
    img[:, 20:] = cv2.cvtColor(np.array([[[110, 200, 200]]], np.uint8),
                               cv2.COLOR_HSV2BGR)[0, 0]  # blue, non-skin

    person = np.zeros((20, 30), bool)
    person[:, :20] = True                        # covers both skin regions

    out = scm._hsv_skin_filter(img, person)
    assert out[:, :20].all(), "skin pixels inside person mask kept"
    assert not out[:, 20:].any(), "non-skin blue (and off-mask) dropped"


# --- _sample_cheek_lab_from_bgr ---

def test_sample_cheek_lab_matches_patch():
    skin = _skin_bgr()
    img = np.tile(skin, (100, 100, 1)).astype(np.uint8)
    bbox = (10, 10, 90, 90)
    L, a, b = scm._sample_cheek_lab_from_bgr(img, bbox)
    exp = scm._bgr_to_lab(skin.reshape(1, 1, 3)).reshape(3)
    assert abs(L - exp[0]) < 2.0 and abs(a - exp[1]) < 2.0 and abs(b - exp[2]) < 2.0


# --- face-inclusion regression guard ---

def test_face_exclusion_removed():
    assert not hasattr(scm, "_face_exclusion_mask"), \
        "face-inclusion: _face_exclusion_mask must stay removed"
    assert not hasattr(scm, "_FACE_EXCLUSION_DILATE"), \
        "face-inclusion: _FACE_EXCLUSION_DILATE must stay removed"
