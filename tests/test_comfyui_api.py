import sys
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from comfyui_api import inject_workflow_values

def test_inject_workflow_values_basic():
    workflow = {
        "1": {
            "inputs": {"text": "original"},
            "_meta": {"title": "CLIP Text Encode"}
        }
    }
    overrides = {
        "CLIP Text Encode": {"inputs.text": "new prompt"}
    }

    patched = inject_workflow_values(workflow, overrides)

    assert patched["1"]["inputs"]["text"] == "new prompt"
    assert workflow["1"]["inputs"]["text"] == "original"
    assert patched is not workflow

def test_inject_workflow_values_multiple_nodes():
    workflow = {
        "1": {
            "inputs": {"text": "original 1"},
            "_meta": {"title": "Node 1"}
        },
        "2": {
            "inputs": {"seed": 123},
            "_meta": {"title": "Node 2"}
        }
    }
    overrides = {
        "Node 1": {"inputs.text": "new 1"},
        "Node 2": {"inputs.seed": 456}
    }

    patched = inject_workflow_values(workflow, overrides)

    assert patched["1"]["inputs"]["text"] == "new 1"
    assert patched["2"]["inputs"]["seed"] == 456
    assert workflow["1"]["inputs"]["text"] == "original 1"
    assert workflow["2"]["inputs"]["seed"] == 123

def test_inject_workflow_values_no_match():
    workflow = {
        "1": {
            "inputs": {"text": "original"},
            "_meta": {"title": "Other Node"}
        }
    }
    overrides = {
        "CLIP Text Encode": {"inputs.text": "new prompt"}
    }

    patched = inject_workflow_values(workflow, overrides)

    assert patched["1"]["inputs"]["text"] == "original"
    # We now add a title cache key, so simple equality fails.
    # We strip it for the comparison.
    patched_no_cache = patched.copy()
    patched_no_cache.pop("_claude_title_cache", None)
    assert patched_no_cache == workflow
    assert patched is not workflow

def test_inject_workflow_values_deep_nesting():
    workflow = {
        "1": {
            "inputs": {
                "nested": {
                    "field": "old"
                }
            },
            "_meta": {"title": "Nested Node"}
        }
    }
    overrides = {
        "Nested Node": {"inputs.nested.field": "new"}
    }

    patched = inject_workflow_values(workflow, overrides)

    assert patched["1"]["inputs"]["nested"]["field"] == "new"
    assert workflow["1"]["inputs"]["nested"]["field"] == "old"


def test_upload_image_caching(tmp_path):
    from unittest.mock import MagicMock
    from comfyui_api import ComfyUIClient

    # Create a dummy image
    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"fake image data")

    client = ComfyUIClient()
    client.session.post = MagicMock()
    client.session.post.return_value.json.return_value = {"name": "remote_test.png"}
    client.session.post.return_value.status_code = 200

    # First upload
    name1 = client.upload_image(str(img_path))
    assert name1 == "remote_test.png"
    assert client.session.post.call_count == 1

    # Second upload (should be cached)
    name2 = client.upload_image(str(img_path))
    assert name2 == "remote_test.png"
    assert client.session.post.call_count == 1

    # Modify file mtime to invalidate cache
    import time
    import os
    new_mtime = img_path.stat().st_mtime + 10
    os.utime(img_path, (new_mtime, new_mtime))

    # Third upload (should re-upload)
    name3 = client.upload_image(str(img_path))
    assert name3 == "remote_test.png"
    assert client.session.post.call_count == 2

