import sys
import json
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from presets import load_preset

def test_load_preset_basic():
    character = "ananya"
    # Find a valid preset name
    with open(ROOT / "character" / character / "presets.yaml", "r") as f:
        data = yaml.safe_load(f)
    preset_name = list(data.keys())[0]

    preset = load_preset(character, preset_name)
    assert isinstance(preset, dict)
    assert "kind" in preset
    assert "overrides" in preset
    assert "defaults" in preset

def test_load_preset_isolation():
    character = "ananya"
    with open(ROOT / "character" / character / "presets.yaml", "r") as f:
        data = yaml.safe_load(f)
    preset_name = list(data.keys())[0]

    p1 = load_preset(character, preset_name)
    orig_kind = p1.get("kind")
    p1["kind"] = "MODIFIED"

    p2 = load_preset(character, preset_name)
    assert p2["kind"] == orig_kind
    assert p2 is not p1

def test_load_preset_not_found():
    with pytest.raises(KeyError):
        load_preset("ananya", "non_existent_preset_12345")

def test_load_preset_char_not_found():
    with pytest.raises(FileNotFoundError):
        load_preset("non_existent_character_12345", "some_preset")
