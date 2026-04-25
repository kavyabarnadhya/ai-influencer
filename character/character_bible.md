# KaviB — Character Bible

## Identity

| Field | Value |
|-------|-------|
| Name | KaviB |
| Age | **23** (always state exactly 23 in all captions, bios, and prompts — never change) |
| Trigger word | `KaviB` (must be first token in every prompt — `keep_tokens=1` in Kohya) |
| Persona | Lifestyle creator, traveller, wellness enthusiast |

## Physical Description (encoded in LoRA — never re-prompt these)

- Short-haired young south asian woman
- Dark eyes, warm brown skin tone
- Petite frame, natural features

**Isolation Rule:** Never describe face shape, eye color, hair color/length, skin tone, or ethnicity in prompts. The LoRA encodes these permanently. Re-prompting anatomy causes "double-baking" (plastic, distorted faces).

## What to Prompt (clothing, setting, pose, mood only)

- Clothing: outfit style, fabric, color
- Setting: location, time of day, environment
- Pose: body position, action, gesture
- Lighting: natural, studio, golden hour, moody
- Mood: expression, vibe, energy

## Content Tiers

| Tier | Platforms | Output Folder | Consent Flag |
|------|-----------|--------------|-------------|
| Lifestyle | Instagram, all public | `output/YYYY-MM-DD/lifestyle/` | None required |
| Seductive | Fanvue, Telegram | `output/private/YYYY-MM-DD/` | `--adult-consent-confirmed` |
| Explicit | Fanvue only | `output/private/YYYY-MM-DD/` | `--adult-consent-confirmed` |

## Disclosure Requirements

- Instagram: every post must include `#AI` in caption
- Fanvue: AI creator label must be enabled on profile
- Telegram: channel description must note AI-generated content

## Training Dataset Specs

- Resolution: 1024×1024 PNG
- Split: 8 closeup + 8 medium shot + 8 full body = 24 images minimum
- Captioning: describe everything *except* permanent physical identity (Isolation Rule)
- Format: one `.txt` file per image, same filename

## Platform ToS Summary

**Instagram:** No explicit nudity. AI-generated content must be disclosed. Age representation must be accurate.

**Fanvue:** Adult content permitted. Requires age verification of creator. AI disclosure required. All depicted individuals must be 18+ (KaviB is always 23).

**Telegram:** Adult channels require age gate. No CSAM. Standard content moderation applies.

## Legal Notes

- Training data: no real-person likenesses without explicit written consent
- Juggernaut XL license: verify commercial use terms before monetization
- KaviB is a fully synthetic persona — no real person is depicted
- Maintain records that training images are AI-generated (bootstrap_seeds.py output)
