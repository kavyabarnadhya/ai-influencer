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
    assert patched == workflow
    assert patched is not workflow # Current implementation always deep copies

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
