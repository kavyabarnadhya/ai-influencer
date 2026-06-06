"""Tests for scripts/lint_carousel_prompts.py — the pre-GPU carousel prompt linter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import lint_carousel_prompts as lint  # noqa: E402


def _e(text):
    return lint.lint_text(text)[0]  # errors


def _w(text):
    return lint.lint_text(text)[1]  # warnings


# --- hand at waist on light garment ---

def test_hand_at_waist_on_white_dress_errors():
    errs = _e("anchor=default | left hand at the waist, white bodycon midi dress, club")
    assert any("LIGHT fabric" in e for e in errs)


def test_hand_at_waist_on_dark_dress_ok():
    # black/navy garment → not a light-fabric fusion risk
    assert not _e("anchor=default | left hand at the waist, navy strapless gown, lounge")


def test_white_in_background_not_flagged():
    # "white pillars" is BG, not the garment → must NOT trip the light-fabric rule
    assert not _e("anchor=default | hand resting on thigh, black metal mini skirt, white pillars behind")


def test_negated_hand_at_waist_not_flagged():
    # "NO hand at the waist" is an instruction to keep hands away → not a violation
    assert not _e("anchor=default | hand on the railing, NO hand at the waist, white bodycon dress")


# --- thin prop on detail slide ---

def test_thin_prop_on_detail_errors():
    errs = _e("anchor=default | tight crop head out, both hands holding ONE wine glass, club")
    assert any("thin held prop" in e for e in errs)


def test_glass_panel_fixture_not_flagged_as_prop():
    # architectural "glass panel" in BG must NOT trip the held-prop rule on a detail slide
    assert not _e(
        "anchor=default | tight crop head out, the bodice, dark wood railing with glass panel behind, "
        "NO face NO chin, NO hands"
    )


def test_detail_no_prop_no_hands_ok():
    assert not _e(
        "anchor=default | extreme tight crop head cropped out, NO hands NO glass, "
        "the gold O-ring detail, NO face NO chin in frame"
    )


# --- object singular guard = warning ---

def test_object_without_guard_warns():
    assert any("singular guard" in w for w in _w("anchor=default | holding a handbag, walking"))


def test_object_with_guard_clean():
    assert not _w("anchor=default | holding ONE single handbag exactly one NOT two NOT duplicate")


# --- §8 forbidden patterns ---

def test_back_to_camera_errors():
    assert any("back-to-camera" in e for e in _e("anchor=default | body turned away from camera, red dress"))


def test_back_to_camera_ok_when_faceswap_false():
    # intentional faceless walk-away is allowed
    assert not _e("faceswap=false | back to camera walking away down the hall, red dress")


def test_hand_on_closure_errors():
    assert any("closure" in e for e in _e("anchor=default | right hand touching the ribbon tie at neckline"))


def test_mirror_errors():
    assert any("portal" in e for e in _e("anchor=default | posing in front of a gold-framed mirror"))


def test_head_out_without_no_face_warns():
    assert any("paint a" in w for w in _w("anchor=default | tight crop collarbone down, the bodice, editorial"))


def test_clean_prompt_no_findings():
    e, w = lint.lint_text(
        "anchor=default | chest-up portrait framing showing face neck shoulders and neckline only, "
        "hand to cheek with fingers near jawline, navy gown, warm bokeh"
    )
    assert not e and not w
