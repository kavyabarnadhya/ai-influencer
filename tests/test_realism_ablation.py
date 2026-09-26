"""GPU-free tests for the one-factor realism matrix and native-pixel sheet."""
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import realism_ablation as ab  # noqa: E402
from comfyui_api import load_workflow


def test_cases_change_one_axis():
    baseline, *variants = ab.cases()
    assert len(variants) == 9
    for case in variants:
        assert sum(getattr(case, field) != getattr(baseline, field)
                   for field in ("swap", "restore", "lora", "cfg")) == 1


def test_workflow_axes_and_secondary_samplers():
    config = {"generation": {"width": 896, "height": 1152, "steps": 30, "negative_prompt": "negative"},
              "characters": {"ananya": {"lora": "Ananya.safetensors"}},
              "models": {"checkpoint": "Juggernaut.safetensors"}}
    template = load_workflow(str(ab.ROOT / "workflows/t2i_sdxl_lora.json"))
    changed = ab.generation_workflow(template, config, "AnanyaAI, silk dress", 42, ab.Case("test", cfg=5, lora=.65))
    assert changed["2"]["inputs"]["strength_model"] == .65
    assert changed["2"]["inputs"]["strength_clip"] == .65
    assert changed["6"]["inputs"]["seed"] == 42
    assert changed["6"]["inputs"]["cfg"] == 5
    for key in ("8", "12"):
        assert changed[key]["inputs"]["seed"] == 42
        assert changed[key]["inputs"]["cfg"] == 5
    assert template["2"]["inputs"]["strength_model"] == .85
    assert template["8"]["inputs"]["cfg"] == 7
    assert template["8"]["inputs"]["seed"] == 0


def test_swap_visibility():
    template = load_workflow(str(ab.ROOT / "workflows/faceswap_reactor.json"))
    off = ab.swap_workflow(template, "source.png", "target.png", 0)
    assert off["3"]["inputs"]["face_restore_visibility"] == 0
    assert off["3"]["inputs"]["swap_model"] == "inswapper_128.onnx"
    assert template["3"]["inputs"]["face_restore_visibility"] == 1


def test_sheet_native_crop_and_invalid_crop(tmp_path):
    paths = []
    for i in range(2):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (64, 72), (i * 100, 2, 3)).save(path)
        paths.append((ab.Case(str(i)), path))
    sheet = tmp_path / "sheet.png"
    ab.render_sheet(paths, sheet, (4, 5, 20, 30))
    with Image.open(sheet) as image:
        assert image.size == (40, 84)
        assert image.getpixel((0, 54)) == (0, 2, 3)
        assert image.getpixel((20, 54)) == (100, 2, 3)
    with pytest.raises(ValueError, match="outside"):
        ab.render_sheet(paths, sheet, (60, 0, 20, 30))
