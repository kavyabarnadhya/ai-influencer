# Character Bible — KaviB

## Persona Overview

| Field | Value |
|-------|-------|
| Name | KaviB |
| Age | **Always stated as 23** in all captions, bios, and posts |
| Trigger word | `KaviB` (first token in every prompt — never omit) |
| LoRA file | `KaviB_v1_Prod.safetensors` |
| LoRA strength | 0.85 |

## Physical Description (for reference only — do NOT re-prompt these)

The LoRA encodes the character's permanent identity features. Re-prompting anatomy
causes "double-baking" that produces plastic, distorted faces.

- Short hair
- Young woman, early 20s appearance
- Naturally photorealistic skin texture (handled by base prompt)

**Never include these in generation prompts:** face shape, eye color, bone structure,
hair color, hair length, ethnicity descriptors. The trigger word `KaviB` alone activates
the LoRA identity.

## Core Prompt Fragment (`character/prompts/base_prompt.txt`)

```
KaviB, photorealistic photograph, professional camera, shallow depth of field,
natural lighting, hyperrealistic skin texture
```

`generate.py` auto-prepends this from config. You only need to supply the scene description.

## Default Negative Prompt

```
blurry, extra fingers, malformed hands, deformed eyes, bad anatomy,
duplicate person, watermark, text, low quality
```

This is baked into `workflows/t2i_sdxl_lora.json`.

## Content Tiers

### Tier 1 — Lifestyle (Instagram / all platforms)
- Fully clothed, public settings, everyday activities
- Prompt file: `character/prompts/lifestyle_prompts.txt`
- Output dir: `output/YYYY-MM-DD/lifestyle/`
- `batch_generate.py --category lifestyle`

### Tier 2 — Seductive (Fanvue / Telegram only)
- Tasteful lingerie, swimwear, implied nudity — no explicit content
- Prompt file: `character/prompts/intimate_prompts.txt`
- Output dir: `output/private/YYYY-MM-DD/`
- `batch_generate.py --category adult --adult-consent-confirmed`
- Required disclosure: AI creator label on every post

### Tier 3 — Explicit (Fanvue only, after platform verification)
- Only after confirming Fanvue's current AI content policy allows this
- Always subject to Fanvue KYC and AI creator labeling requirements
- **Never post to Instagram** — violates Instagram Community Guidelines

## Seed Image Guide

Training dataset: 24 images, 1024×1024 PNG, 8+8+8 split

| Category | Count | Framing |
|----------|-------|---------|
| closeup | 8 | Chin to forehead — skin pores and iris detail |
| medium | 8 | Waist up — shoulder width and hair-to-body ratio |
| fullbody | 8 | Head to toe — height relative to background |

See `setup/train_lora_guide.md` for full dataset preparation instructions.

## Platform ToS Summary

| Platform | AI creators? | Adult content? | Disclosure required? |
|----------|-------------|---------------|---------------------|
| Instagram | Yes | No | Yes — `#AI` |
| Fanvue | Yes (AI creator label) | Yes (platform policy) | Yes — AI label |
| Telegram | Yes | Yes (channel rules) | Recommended |

## Legal & Ethical Requirements

- **Age representation**: KaviB is always stated as 23. Never depict, caption, or imply
  any character is under 18. This is non-negotiable.
- **Training data**: No real-person likenesses unless you have explicit written consent
  from the depicted individual. Never use celebrity, influencer, or scraped third-party faces.
- **Disclosure**: All public posts must be labeled as AI-generated.
- **Model licenses**: Juggernaut XL model card states "not permitted behind API services."
  Contact RunDiffusion for commercial licensing before monetizing at scale.
- **No deepfakes**: This pipeline must never be used to generate realistic images of
  real, identifiable individuals without their explicit consent.
