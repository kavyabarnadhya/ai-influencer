"""One-off: generate D3 slide_04 (rear view) via Kontext image-editing chained
directly off the APPROVED slide_03 output, not the carousel anchor.

This is item (3) of the D3 pipeline-hardening pass: rear/alternate angles
derived from the approved front reference via image conditioning, not an
independent text-only generation. Reuses faceswap_carousel.py's own
_inject_flux_kontext + ComfyUIClient rather than inventing new plumbing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from comfyui_api import ComfyUIClient, find_comfyui_port, load_workflow
from faceswap_carousel import _inject_flux_kontext

ROOT = Path(__file__).parent.parent
SOURCE = ROOT / "output/2026-09-29/ananya/d3_coffee_run_outfit_math/slide_03.png"
OUT = ROOT / "output/2026-09-27/ananya/d3_slide04_from_slide03.png"

PROMPT = (
    "same woman WEARING a light-wash blue denim jacket open over her cream ribbed sweater, "
    "denim jacket falling to natural hip length, denim jacket visible from behind across both "
    "shoulders and arms, black joggers, one small tan leather crossbody bag with a rectangular "
    "flap and a gold buckle clasp and gold chain-link strap hardware, worn over the denim jacket, "
    "keeping exact same tan leather crossbody bag shape clasp color and scale, cream baseball cap "
    "worn forward, DO NOT remove the denim jacket, denim jacket is the outermost visible layer "
    "from behind, captured from behind walking slowly toward the painted cafe shopfront, "
    "mid-stride one foot forward, body and head facing away face NOT visible, both arms relaxed "
    "at sides showing denim jacket sleeves, bright morning daylight"
)

port = find_comfyui_port()
if port is None:
    raise SystemExit("ComfyUI not reachable")
client = ComfyUIClient("127.0.0.1", port)

uploaded = client.upload_image(str(SOURCE))
template = load_workflow(str(ROOT / "workflows" / "flux_kontext.json"))
wf = _inject_flux_kontext(template, PROMPT, uploaded, seed=987654321, bg_lock=True)

prompt_id = client.submit_workflow(wf)
outputs = client.wait_for_completion(prompt_id, timeout=180)
if len(outputs) != 1:
    raise SystemExit(f"Expected one image, got {len(outputs)}")
ref = outputs[0]
data = client.download_image(ref["filename"], ref.get("subfolder", ""), ref.get("type", "output"))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_bytes(data)
print(f"Saved: {OUT}")
