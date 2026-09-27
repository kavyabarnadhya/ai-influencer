# D3 pipeline-hardening pass — findings and what was implemented

Scope: PR #161's slides 02-04 (cumulative outfit continuity: jacket, bag
shape/clasp/color/scale). Four safeguards were requested; here's what's
actually available in this ComfyUI install and what was built.

## (1) IP-Adapter / reference conditioning for every slide

**Constraint, stated plainly:** the only IPAdapter model file present is
`models/ipadapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors`
— a **face-only** IPAdapter for SDXL. There is no general/composition or
garment-reference IPAdapter model, and no FLUX IPAdapter model, on disk.
The IPAdapter *node classes* (`IPAdapterAdvanced`, `IPAdapterEmbeds`, etc.)
exist because the custom node pack is installed, but a node class without
a matching model file is not usable — same category of gap as the img2vid
model check from an earlier session. A face IPAdapter is also redundant
here: identity is already locked by ReActor faceswap against
`face_ref_v2.png`, and a garment/bag isn't a face.

**What's actually already doing image conditioning:** `faceswap_carousel.py`'s
`--kontext` flag (already in use for every D3 slide) feeds a real reference
image — the anchor — into FLUX Kontext dev as edit-conditioning for every
slide. That's genuine image-to-image conditioning, not independent
text-only generation; it was already satisfying most of the intent behind
this item before this pass started. The earlier jacket-dropping defect was
a prompt-strength problem (Kontext's text delta overriding the image
condition), not a missing-conditioning problem — fixed by making the
outfit-lock language explicit and repeated (see item 2), not by adding new
machinery.

**Verdict:** implemented with what exists (Kontext image-conditioning);
true garment-reference IPAdapter is not available without downloading a
new model file, which wasn't done (no new paid/unvetted downloads without
asking).

## (2) Shared verbatim prompt block

Implemented. `character/ananya/carousel_prompts/_d3_cumulative_fix_v3.txt`
carries one literal block (bag shape/clasp/color/scale, jacket length,
jewelry) copy-pasted unchanged into slide_02 and slide_03's prompts — only
pose/angle/accessory-addition language varies. Documented at the top of
that file so future edits can verify they're keeping the block verbatim
rather than paraphrasing it per-slide (paraphrasing was the root cause of
the first-round jacket disappearance).

## (3) Rear/alternate angle via img2img off the approved front reference

Implemented. `scripts/_d3_slide04_from_slide03.py` uploads the **approved
slide_03 output** (not the carousel anchor) as the Kontext init image and
derives slide_04 from it directly, reusing `faceswap_carousel.py`'s own
`_inject_flux_kontext()` function and `comfyui_api.py`'s client — no new
plumbing invented. Confirmed via ComfyUI's `/history` API that the
executed workflow's `_claude_inject_init_image` node loaded `slide_03.png`
verbatim, not the anchor.

**Known side-effect, disclosed:** FLUX Kontext's own `FluxKontextImageScale`
node re-scales the input to its preferred aspect ratio internally, so the
raw output came back at 752x1392 instead of the pipeline's standard
1080x1920. Corrected with a plain resize+center-crop back to 1080x1920 to
match the other 4 slides — this is post-hoc image processing, not a second
generation pass, and was visually re-checked after the crop to confirm no
important content (jacket, bag, cap) was cut off.

**Not done:** slide_04 did not go through the full
hand-detail/skin-lock post-process stages that `faceswap_carousel.py`'s
own pipeline normally runs (Stage 3.5/3.6), because this slide is faceless
and hands are not prominently in frame — the marginal quality gain wasn't
worth the added generation time given the "avoid wasteful broad reruns"
instruction. Flagging this as a real, intentional gap: if hand quality on
this slide becomes a concern on closer review, it can be run through
`reprocess_carousel_post.py` separately.

## (4) Automated cross-slide vision QA gate

Implemented: `scripts/qa_accessory_continuity.py`. Crops a fixed
lower-torso/bag region from a canonical slide and each slide to compare,
computes CLIP (ViT-L-14, openai weights — same model already used by
`clip_similarity_audit.py`, no new download) embedding cosine similarity,
and exits non-zero if any comparison falls below threshold (default 0.75).

**Limitation, stated plainly:** this is a coarse whole-region embedding
comparison. It reliably catches gross mismatches — wrong bag color, jacket
missing entirely, wrong garment category — because those produce a large
embedding shift. It does **not** verify precise clasp geometry, exact hue
matching, or subtle shape differences at a fine-grained level; CLIP's
embedding space isn't precise enough for that. It is a pre-review gate
that can catch the class of defect that shipped in the first PR #161
attempt (jacket present vs. absent), not a replacement for the
full-resolution human review this repo's `carousel_workflow.md` already
mandates. The script prints this caveat every time it runs, pass or fail,
so it can't be mistaken for a stronger guarantee than it is.

Run against the final D3 set (slide_02 as canonical, comparing slide_03 and
slide_04): **PASSED**, 0.9635 and 0.9432 cosine similarity respectively
(well above the 0.75 threshold), consistent with the visual QC.

## Bottom line

Items (2), (3), (4) implemented with real, working code using this
pipeline's existing mechanisms. Item (1) as literally specified
(IPAdapter-based whole-outfit reference conditioning) is not available on
this machine without downloading a new model — reported rather than faked;
the existing Kontext image-conditioning already in use covers the same
underlying need for this fix.
