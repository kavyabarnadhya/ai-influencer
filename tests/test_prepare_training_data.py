import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_training_data as prep


def test_flux_forbidden_terms_flag_identity_descriptors():
    caption = (
        "waist-up portrait of AnanyaAI, seen from the front at eye level, "
        "with loose hair, wearing a black blazer. Brown skin, dark eyes, north indian woman."
    )

    found = prep.find_forbidden_caption_terms(caption)

    assert "skin tone" in found
    assert "eye identity" in found
    assert "ethnicity" in found


def test_flux_forbidden_terms_allow_variable_controls():
    caption = (
        "waist-up portrait of AnanyaAI, seen from a three-quarter angle at eye level, "
        "with loose dark hair, wearing a black linen blazer and tiny gold earrings. "
        "She is leaning on a cafe table with a soft smile. Warm window light. Timber cafe background."
    )

    assert prep.find_forbidden_caption_terms(caption) == []


def test_infer_flux_shot_type_uses_filename_before_dimensions(tmp_path):
    image_path = tmp_path / "12_fullbody_candidate.png"
    image_path.write_bytes(b"not a real image")

    assert prep.infer_flux_shot_type(image_path) == "full-body portrait"


def test_choose_layout_prefers_flux_when_training_data_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    training_dir = tmp_path / "character" / "ananya" / "training_data"
    training_dir.mkdir(parents=True)
    (training_dir / "01.png").write_bytes(b"placeholder")

    char_cfg = {
        "seeds_dir": "character/ananya/seeds",
        "training_data_dir": "character/ananya/training_data",
    }

    assert prep.choose_layout(char_cfg, "auto") == "flux"


def test_has_flux_images_non_existent():
    assert prep.has_flux_images(Path("non_existent_directory_12345")) is False


def test_has_flux_images_empty(tmp_path):
    assert prep.has_flux_images(tmp_path) is False


def test_has_flux_images_with_supported_image(tmp_path):
    (tmp_path / "image.png").write_bytes(b"placeholder")
    assert prep.has_flux_images(tmp_path) is True


def test_has_flux_images_with_only_unsupported_files(tmp_path):
    (tmp_path / "notes.txt").write_text("AnanyaAI draft caption")
    (tmp_path / "image.bmp").write_bytes(b"placeholder")
    assert prep.has_flux_images(tmp_path) is False


def test_choose_layout_prefers_sdxl_when_no_flux_images(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    training_dir = tmp_path / "character" / "ananya" / "training_data"
    training_dir.mkdir(parents=True)
    (training_dir / "notes.txt").write_text("no images here")

    char_cfg = {
        "seeds_dir": "character/ananya/seeds",
        "training_data_dir": "character/ananya/training_data",
    }

    assert prep.choose_layout(char_cfg, "auto") == "sdxl"
