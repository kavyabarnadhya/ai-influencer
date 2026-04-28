# Ananya FLUX LoRA and Reels Pipeline

## Goal

Train an Ananya FLUX image LoRA first, then use it to create consistent 9:16 anchor stills for cloud image-to-video reels. The FLUX LoRA is not a Wan/LTX video LoRA, but it gives the video model stronger starting frames.

## Dataset Standard

- Target 25 excellent images in `character/ananya/training_data/`.
- Keep only images with strong Ananya identity, clean eyes, clean mouth, believable skin, and no hand or face artifacts.
- Prefer variety across close-up, waist-up, and full-body framing; front, three-quarter, and profile angles; indoor, outdoor, daylight, night, and mixed lighting.
- Include Indian fashion range: saree, salwar/kurta, and Indo-western looks.
- Include a controlled premium non-nude range: low necklines, short dresses, thigh-visible poses, elegant sensual expressions, and hotel/lounge/rooftop evening styling.
- Keep premium images to roughly 20-30% of the final set so the LoRA learns the range without making every prompt look like premium content.
- Preserve useful mixed aspect ratios. The ai-toolkit FLUX trainer buckets and downscales images, so do not force square crops unless the image needs cropping for quality.
- Reject watermarks, compression artifacts, duplicate outfits/backgrounds, blurry faces, and images with obvious identity drift.
- Reject nudity, lingerie-only, explicit sexual acts, fetish framing, pornographic poses, or anything that would make the identity LoRA overfit toward NSFW output.

Suggested 25-image mix:

- 6 close-up identity anchors
- 5 Indian fashion looks: saree, salwar/kurta, Indo-western
- 5 public lifestyle/fashion looks
- 5 premium non-nude glamour looks
- 4 angle/lighting stress tests: profile, high/low camera, night/neon, harsh sun

## Caption Rules

Every image must have a matching `.txt` caption beside it.

Use the face-fixed boundary:

- Include `AnanyaAI` in every caption.
- Do not caption permanent facial identity: face shape, skin tone, ethnicity, eye color, age, body type.
- Do caption variable controls: hair styling/color, outfit, makeup, accessories, expression, pose, action, camera angle, lighting, and background.
- For premium looks, caption visible clothing and styling plainly, such as `tasteful low neckline`, `short cocktail dress`, `legs visible`, `confident seductive expression`, or `evening lounge styling`.
- Use short natural-language captions, not long comma-tag dumps.
- Avoid negative phrasing like `no hands` or `without jewelry`; caption what is visible.

Template:

```text
<shot type> of AnanyaAI, seen from <angle> at <camera height>, with <hair style/color>, wearing <outfit/accessories>. She is <pose/action> with <expression>. <lighting>. <short background>.
```

## Local Commands

Generate editable FLUX caption scaffolds:

```powershell
python scripts\prepare_training_data.py --character ananya --caption-style flux
```

Validate the FLUX dataset:

```powershell
python scripts\prepare_training_data.py --character ananya --layout flux --validate
```

Package the FLUX dataset for RunPod:

```powershell
python scripts\prepare_training_data.py --character ananya --layout flux --zip-only
```

The package includes `ai_toolkit_flux_lora_rank16.yaml` with the first-run baseline:

- trainer: `ostris/ai-toolkit`
- base model: `black-forest-labs/FLUX.1-dev`
- network: `linear: 16`, `linear_alpha: 16`
- `caption_dropout_rate: 0`
- `shuffle_tokens: false`

## Reels Lane

After the FLUX LoRA is trained and copied to ComfyUI as `AnanyaAI_FLUX_v1.safetensors`, use `--workflow flux_schnell_lora` for Ananya anchor stills. The original `flux_schnell` workflow stays LoRA-free for background or text-only experiments.

Generate 9:16 anchor stills and animate them in a cloud image-to-video tool such as Wan/LTX/Comfy Cloud/fal/Replicate.

```powershell
python scripts\generate.py --workflow flux_schnell_lora --reel-anchor --prompt "waist-up cafe reel anchor, slow smile setup, white linen blazer, warm window light"
```

Anchor stills are saved under `output/YYYY-MM-DD/ananya/reels/anchors/`.

Start with low-risk believable motions:

- slow head turn
- soft smile
- walking slowly
- coffee sip
- hair movement
- camera push-in
- phone mirror pose

Reject video clips with face drift, eye flicker, melted hands, warped jewelry, or unnatural motion. Defer a separate Wan/LTX video LoRA until image-to-video identity drift becomes a real growth bottleneck.
