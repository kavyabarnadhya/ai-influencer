"""D3 QC correction: regenerate slide_02 via img2img chained off the APPROVED
slide_03 (canonical per user review), not the anchor. slide_02's chunky gold
chain strap + layered necklace didn't match slide_03/04's leather strap +
single-pendant necklace, even though CLIP crop similarity scored high (the
gross color palette was similar; strap material/necklace style were not
caught by that coarse check). Deriving slide_02 from slide_03 pixel-for-pixel
(three-quarter angle instead of full-face) is the same mechanism that
correctly held the strap/necklace for slide_04.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from comfyui_api import ComfyUIClient, find_comfyui_port, load_workflow
from faceswap_carousel import _inject_flux_kontext

ROOT = Path(__file__).parent.parent
SOURCE = ROOT / "output/2026-09-29/ananya/d3_coffee_run_outfit_math/slide_03.png"
OUT = ROOT / "output/2026-09-27/ananya/d3_slide02_from_slide03.png"

PROMPT = (
    "same woman WEARING a light-wash blue denim jacket open over her cream ribbed sweater, "
    "denim jacket falling to natural hip length, denim jacket sleeves visible on both arms, "
    "denim jacket lapels and collar clearly visible framing the sweater, black joggers, "
    "one small tan leather crossbody bag with a rectangular flap and a gold buckle clasp, "
    "bag strap is the SAME tan leather as the bag body NOT a metal chain NOT gold links, "
    "bag worn across the chest over the denim jacket, "
    "one single thin delicate gold chain necklace with one small rectangular pendant, "
    "NOT layered NOT double chain NOT a square pendant, small gold stud earrings, "
    "keeping exact same tan leather crossbody bag shape clasp color scale and leather strap, "
    "DO NOT remove the denim jacket, denim jacket is the outermost visible layer, "
    "removing the sunglasses and cap, three-quarter angle facing slightly left toward camera, "
    "weight on left leg, right hand resting on the bag strap, left arm relaxed at side showing "
    "the denim jacket sleeve, direct camera gaze, warm smile, soft morning light"
)

port = find_comfyui_port()
if port is None:
    raise SystemExit("ComfyUI not reachable")
client = ComfyUIClient("127.0.0.1", port)

uploaded = client.upload_image(str(SOURCE))
template = load_workflow(str(ROOT / "workflows" / "flux_kontext.json"))
wf = _inject_flux_kontext(template, PROMPT, uploaded, seed=192837465, bg_lock=True)

prompt_id = client.submit_workflow(wf)
print(f"submitted: {prompt_id}")
outputs = client.wait_for_completion(prompt_id, timeout=170)
if len(outputs) != 1:
    raise SystemExit(f"Expected one image, got {len(outputs)}")
ref = outputs[0]
data = client.download_image(ref["filename"], ref.get("subfolder", ""), ref.get("type", "output"))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_bytes(data)
print(f"Saved: {OUT}")
