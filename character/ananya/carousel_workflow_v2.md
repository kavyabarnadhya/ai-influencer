# Ananya Carousel Workflow V2

Reference for future coding agents before changing or running carousel generation.

Last updated: 2026-05-05

## Goal

Instagram carousels must feel like a mini photoshoot:

- Same outfit, hair, accessories, body structure, scene, and ambiance across model slides
- Different poses, camera angles, and shot types: full body, medium/three-quarter, portrait/bust
- Ambient/no-model final slide when useful
- No bad hands; hand artifacts are hard rejects

## Current Production Settings

Use `scripts/generate_carousel.py` with:

- `--poses-dir character/ananya/poses/carousel_production_v2`
- `--candidates 3`
- Face reference: `character/ananya/reference_board/face_ref_001_2890463320.png`
- Checkpoint: `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors`
- LoRA: `AnanyaAI_v1_Prod.safetensors` at `0.75`
- IPAdapter at `0.35`

Role-specific generation settings in `config.yaml`:

- `medium`: img2img denoise `0.74`, ControlNet `0.75`
- `close`: img2img denoise `0.82`, ControlNet `0.8`
- `wide`: t2i anchor, ControlNet `0.75`

The workflow intentionally combines img2img and ControlNet:

- Slide 1 establishes outfit/color/style as the visual anchor.
- Later model slides use img2img for styling consistency and ControlNet for pose/angle variation.
- Ambient slides use plain `t2i_sdxl_lora.json` with no face/pose reference.

## Body Structure Policy

Ananya should read as realistic M-size, not lean/slim.

Use the current body token:

`realistic size M Indian woman, medium curvy build, fuller hips and thighs, natural waist, soft stomach, fuller bust, realistic proportions`

Avoid relying on positive prompt phrases such as `not skinny`; slim-body terms are handled in the global negative prompt.

## Candidate Selection Policy

Always review generated candidates manually before calling a carousel postable.

Required selected set:

- One clean wide/full-body shot
- One clean medium/three-quarter shot with a visibly different angle
- One clean close/portrait/bust shot
- Optional ambient slide with no person

Reject candidates with:

- Double/deformed hands
- Outfit color or silhouette drift
- Wrong hair style
- Overly lean body structure
- Face identity drift
- Ambient slide containing a person

## Calibration Findings

Run: `output/2026-05-05/ananya/carousel_calibration_bodycon_dusk_v2_01`

Selected set:

`output/2026-05-05/ananya/carousel_calibration_bodycon_dusk_v2_01/selected_best/`

Important lessons:

- V2 candidate generation produced meaningful full/medium/close angle variation.
- `slide_2_medium_cand_2_290916075.png` had a double/deformed hand and was rejected.
- Its source pose was moved to `character/ananya/poses/carousel_production_v2/rejected/medium_02_double_hand.png`.
- Close candidates 1 and 2 drifted into off-shoulder styling; close candidate 3 held the short-sleeve bodycon better.
- The body still looked slightly lean until body tokens and base prompt were updated to the M-size policy above.

## Typical Command

```powershell
$env:UV_CACHE_DIR='.uv-cache'; uv run python scripts\generate_carousel.py `
  --scene "soft dusk rooftop, sharp city background, warm ambient terrace lights, Mumbai skyline" `
  --outfit "berry pink fitted bodycon dress, scoop neck, short sleeves, above-knee hem, fitted through hips and thighs" `
  --hair "loose waves, side parted, natural" `
  --slides 4 --name my_carousel `
  --face-ref "character/ananya/reference_board/face_ref_001_2890463320.png" `
  --poses-dir "character/ananya/poses/carousel_production_v2" `
  --candidates 3
```

## Required Companion References

Before writing prompts or changing carousel generation, read:

- `character/ananya/prompt_cookbook.md`
- `character/ananya/fashion_research.md`
- `config.yaml`
