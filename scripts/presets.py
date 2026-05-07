from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


def load_preset(character: str, preset_name: str) -> dict:
    path = ROOT / "character" / character / "presets.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No presets file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if preset_name not in data:
        raise KeyError(
            f"Preset '{preset_name}' not in {path}. Available: {sorted(data.keys())}"
        )
    preset = data[preset_name]
    preset.setdefault("kind", "single")
    preset.setdefault("overrides", {})
    preset.setdefault("defaults", {})
    return preset


def render_prompt(template: str, *, outfit: str, hair: str, scene: str) -> str:
    fields = defaultdict(str, outfit=outfit or "", hair=hair or "", scene=scene or "")
    return template.format_map(fields)


def merge_singleshot(preset: dict, cli: dict) -> dict:
    """CLI wins over preset. Returns flat resolved dict."""
    d = preset.get("defaults", {})
    o = preset.get("overrides", {})
    return {
        "workflow": cli.get("workflow") or preset.get("workflow"),
        "outfit":   cli.get("outfit")   or d.get("outfit", ""),
        "hair":     cli.get("hair")     or d.get("hair", ""),
        "scene":    cli.get("scene")    or d.get("scene", ""),
        "pose":     cli.get("pose")     or preset.get("pose"),
        "face_ref": cli.get("image")    or preset.get("face_ref"),
        "steps":    cli.get("steps")    or preset.get("steps"),
        "seed":     cli.get("seed") if cli.get("seed") is not None else preset.get("seed"),
        "lora_strength":       o.get("lora_strength"),
        "ipadapter_strength":  o.get("ipadapter_strength"),
        "ipadapter_start_at":  o.get("ipadapter_start_at"),
        "ipadapter_end_at":    o.get("ipadapter_end_at"),
        "denoise":             o.get("denoise"),
        "controlnet_strength": o.get("controlnet_strength"),
    }
