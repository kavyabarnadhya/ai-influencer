"""
Unit tests for the GPU-free pure logic in scripts/faceswap_carousel.py:
anchor-config validation, FLUX-Kontext prompt injection (incl BG-lock
auto-append), and slide prompt-file parsing. No ComfyUI / GPU / models.

These guard the spots that have produced real carousel failures: anchor YAML
schema mistakes, a missing BG-lock token, and slide-line parsing drift.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import faceswap_carousel as fc  # noqa: E402

_BG_LOCK = ", same background, same scene, unchanged environment"


def _write(tmp_path, text):
    p = tmp_path / "anchor.yaml"
    p.write_text(text, encoding="utf-8")
    return p


# --- load_anchor_config ---

def test_anchor_config_single_valid(tmp_path):
    cfg = fc.load_anchor_config(_write(tmp_path, "anchor_prompt: a woman in a red dress\nanchor_seed: 334521876\n"))
    assert cfg["mode"] == "single"
    assert cfg["anchor_prompt"].startswith("a woman")
    assert cfg["anchor_seed"] == 334521876


def test_anchor_config_multi_valid(tmp_path):
    cfg = fc.load_anchor_config(_write(
        tmp_path,
        "shared_tail: in a red silk dress on a balcony\n"
        "anchors:\n  standing:\n    prompt: a woman standing\n  sitting:\n    prompt: a woman sitting\n",
    ))
    assert cfg["mode"] == "multi"
    assert set(cfg["anchors"]) == {"standing", "sitting"}


def test_anchor_config_rejects_both_schemas(tmp_path):
    with pytest.raises(ValueError, match="cannot have both"):
        fc.load_anchor_config(_write(tmp_path, "anchor_prompt: x\nanchors:\n  g:\n    prompt: y\n"))


def test_anchor_config_rejects_neither_schema(tmp_path):
    with pytest.raises(ValueError, match="must have either"):
        fc.load_anchor_config(_write(tmp_path, "anchor_seed: 1\n"))


def test_anchor_config_rejects_empty_prompt(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        fc.load_anchor_config(_write(tmp_path, "anchor_prompt: '   '\n"))


def test_anchor_config_rejects_non_int_seed(tmp_path):
    with pytest.raises(ValueError, match="anchor_seed.*integer"):
        fc.load_anchor_config(_write(tmp_path, "anchor_prompt: x\nanchor_seed: not_a_number\n"))


def test_anchor_config_body_lora_strength_range(tmp_path):
    # valid
    cfg = fc.load_anchor_config(_write(tmp_path, "anchor_prompt: x\nanchor_body_lora_strength: 0.5\n"))
    assert cfg["anchor_body_lora_strength"] == 0.5
    # out of range
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        fc.load_anchor_config(_write(tmp_path, "anchor_prompt: x\nanchor_body_lora_strength: 1.5\n"))


def test_anchor_config_multi_group_missing_prompt(tmp_path):
    with pytest.raises(ValueError, match="missing required 'prompt'"):
        fc.load_anchor_config(_write(tmp_path, "anchors:\n  g:\n    seed: 1\n"))


# --- lens_profile ---

def test_lens_profile_valid(tmp_path):
    cfg = fc.load_anchor_config(_write(tmp_path, "anchor_prompt: x\nlens_profile: selfie\n"))
    assert cfg["lens_profile"] == "selfie"


def test_lens_profile_rejects_unknown(tmp_path):
    with pytest.raises(ValueError, match="lens_profile.*must be one of"):
        fc.load_anchor_config(_write(tmp_path, "anchor_prompt: x\nlens_profile: dslr\n"))


def test_lens_suffix_unset_is_empty():
    assert fc._lens_suffix({"anchor_prompt": "x"}) == ""
    assert fc._lens_suffix(None) == ""


def test_lens_suffix_selfie_has_deep_focus():
    s = fc._lens_suffix({"lens_profile": "selfie"})
    assert s.startswith(", ")
    assert "NO bokeh" in s and "deep focus" in s


def test_lens_suffix_editorial_has_bokeh():
    s = fc._lens_suffix({"lens_profile": "editorial"})
    assert "bokeh" in s and "depth of field" in s


# --- _inject_flux_kontext (BG-lock auto-append) ---

def _kontext_wf():
    return {
        "1": {"_meta": {"title": "_claude_inject_prompt"}, "inputs": {"text": ""}},
        "2": {"_meta": {"title": "_claude_inject_init_image"}, "inputs": {"image": ""}},
        "3": {"_meta": {"title": "_claude_inject_seed"}, "inputs": {"seed": 0}},
    }


def test_kontext_bg_lock_appended_by_default():
    wf = fc._inject_flux_kontext(_kontext_wf(), "woman in red dress", "anchor.png", 42)
    assert wf["1"]["inputs"]["text"] == "woman in red dress" + _BG_LOCK


def test_kontext_bg_lock_can_be_disabled():
    wf = fc._inject_flux_kontext(_kontext_wf(), "woman in red dress", "anchor.png", 42, bg_lock=False)
    assert wf["1"]["inputs"]["text"] == "woman in red dress"
    assert _BG_LOCK not in wf["1"]["inputs"]["text"]


def test_kontext_injects_image_and_seed():
    wf = fc._inject_flux_kontext(_kontext_wf(), "p", "slide_init.png", 1234)
    assert wf["2"]["inputs"]["image"] == "slide_init.png"
    assert wf["3"]["inputs"]["seed"] == 1234


# --- parse_prompts_file ---

def _prompts(tmp_path, text):
    p = tmp_path / "slides.txt"
    p.write_text(text, encoding="utf-8")
    return fc.parse_prompts_file(p, default_denoise=0.6)


def test_parse_skips_blank_and_comment_lines(tmp_path):
    rows = _prompts(tmp_path, "# header comment\n\n   \nanchor=default | a real slide\n")
    assert len(rows) == 1
    assert rows[0]["prompt"] == "a real slide"


def test_parse_extracts_anchor_token_and_strips_it(tmp_path):
    rows = _prompts(tmp_path, "anchor=standing | woman walking\n")
    assert rows[0]["anchor"] == "standing"
    assert rows[0]["prompt"] == "woman walking"
    assert "anchor=" not in rows[0]["prompt"]


def test_parse_denoise_token_and_default(tmp_path):
    rows = _prompts(tmp_path, "denoise=0.85 | a\nanchor=default | b\n")
    assert rows[0]["denoise"] == 0.85
    assert rows[1]["denoise"] == 0.6  # falls back to default


def test_parse_malformed_denoise_falls_back(tmp_path):
    rows = _prompts(tmp_path, "denoise=notafloat | a\n")
    assert rows[0]["denoise"] == 0.6  # no crash, default used


def test_parse_plain_line_no_tokens(tmp_path):
    rows = _prompts(tmp_path, "just a plain prompt line\n")
    assert rows[0]["prompt"] == "just a plain prompt line"
    assert rows[0]["anchor"] is None
    assert rows[0]["faceswap"] is True  # default: faceswap on


def test_parse_faceswap_false_token(tmp_path):
    rows = _prompts(tmp_path, "faceswap=false | back of head shot\nanchor=default | normal slide\n")
    assert rows[0]["faceswap"] is False
    assert rows[0]["prompt"] == "back of head shot"
    assert "faceswap=" not in rows[0]["prompt"]
    assert rows[1]["faceswap"] is True  # absent token defaults to True


def test_parse_ultra_token_defaults_off(tmp_path):
    p = tmp_path / "slides.txt"
    p.write_text("anchor=default | a\nultra=true | b\nultra=false | c\n", encoding="utf-8")
    rows = fc.parse_prompts_file(p, default_denoise=0.6)  # default_ultra defaults False
    assert [r["ultra"] for r in rows] == [False, True, False]
    assert "ultra=" not in rows[1]["prompt"]


def test_parse_ultra_token_default_on_with_override(tmp_path):
    p = tmp_path / "slides.txt"
    p.write_text("anchor=default | a\nultra=false | b\n", encoding="utf-8")
    rows = fc.parse_prompts_file(p, default_denoise=0.6, default_ultra=True)
    assert rows[0]["ultra"] is True   # inherits global --ultra
    assert rows[1]["ultra"] is False  # per-slide override wins


def test_parse_ultra_numeric_denoise(tmp_path):
    p = tmp_path / "slides.txt"
    p.write_text("ultra=0.44 | a\nultra=true | b\nultra=0 | c\nanchor=default | d\n", encoding="utf-8")
    rows = fc.parse_prompts_file(p, default_denoise=0.6)
    # numeric value turns ultra ON and sets per-slide denoise
    assert rows[0]["ultra"] is True and rows[0]["ultra_denoise"] == 0.44
    # bare true = on, denoise None (workflow default 0.38)
    assert rows[1]["ultra"] is True and rows[1]["ultra_denoise"] is None
    # ultra=0 = off
    assert rows[2]["ultra"] is False and rows[2]["ultra_denoise"] is None
    # absent token = off, no denoise
    assert rows[3]["ultra"] is False and rows[3]["ultra_denoise"] is None
    assert "ultra=" not in rows[0]["prompt"]


def test_parse_cands_token(tmp_path):
    p = tmp_path / "slides.txt"
    p.write_text("cands=3 | a\nanchor=default | b\ncands=notanint | c\n", encoding="utf-8")
    rows = fc.parse_prompts_file(p, default_denoise=0.6)
    assert rows[0]["cands"] == 3          # parsed
    assert rows[1]["cands"] is None       # absent => global default
    assert rows[2]["cands"] is None       # malformed => default, no crash
    assert "cands=" not in rows[0]["prompt"]


def test_parse_cands_clamped(tmp_path):
    p = tmp_path / "slides.txt"
    p.write_text("cands=99 | a\ncands=0 | b\n", encoding="utf-8")
    rows = fc.parse_prompts_file(p, default_denoise=0.6)
    assert rows[0]["cands"] == 8   # clamped to max 8
    assert rows[1]["cands"] == 1   # clamped to min 1


def test_parse_ultra_denoise_clamped(tmp_path):
    p = tmp_path / "slides.txt"
    p.write_text("ultra=5 | a\n", encoding="utf-8")
    rows = fc.parse_prompts_file(p, default_denoise=0.6)
    assert rows[0]["ultra"] is True
    assert rows[0]["ultra_denoise"] == 1.0   # clamped to <=1.0 (denoise>1 crashes sampler)


def test_inject_realism_overrides_denoise():
    wf = {
        "2": {"_meta": {"title": "_claude_inject_input_image"}, "class_type": "LoadImage", "inputs": {"image": ""}},
        "7": {"_meta": {"title": "Subject detail pass (skin/hair/fabric, BG clean)"}, "class_type": "DetailerForEach", "inputs": {"denoise": 0.38}},
    }
    out = fc._inject_realism(wf, "base.png", denoise=0.44, propagate_cache=False)
    assert out["7"]["inputs"]["denoise"] == 0.44
    assert out["2"]["inputs"]["image"] == "base.png"
    # None leaves the workflow default untouched
    wf2 = {
        "2": {"_meta": {"title": "_claude_inject_input_image"}, "class_type": "LoadImage", "inputs": {"image": ""}},
        "7": {"_meta": {"title": "x"}, "class_type": "DetailerForEach", "inputs": {"denoise": 0.38}},
    }
    out2 = fc._inject_realism(wf2, "b.png", denoise=None, propagate_cache=False)
    assert out2["7"]["inputs"]["denoise"] == 0.38
