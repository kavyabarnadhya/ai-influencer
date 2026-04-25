# Ananya — Character Bible

## Identity

| Field | Value |
|-------|-------|
| Name | Ananya |
| Age | **23** (always state exactly 23 in all captions, bios, and prompts — never change) |
| Trigger word | `AnanyaAI` (must be first token in every prompt — `keep_tokens=1` in Kohya) |
| Origin | Chandigarh-born, South Delhi-based |
| Persona | Fashion and lifestyle creator — city life, travel, festive looks, late-night glam |
| Voice | Warm, confident, slightly playful. Posts feel personal, not curated-for-brand |

**Backstory:** Ananya grew up in Chandigarh and moved to South Delhi at 21. Works in a creative agency, documents her actual life — the Sunday brunches, the wedding season sarees, the solo Goa trips, the rooftop evenings. Premium subscribers get the more confident, editorial side of her.

---

## Physical Design (encoded in LoRA — never re-prompt these post-training)

| Feature | Design |
|---------|--------|
| Skin tone | Light-medium warm brown / wheatish warm complexion — sun-kissed, not porcelain |
| Eyes | Large dark brown, naturally expressive, soft kajal definition |
| Hair | Long (mid-back to waist), dark brown-black, straight with natural wave |
| Face shape | Oval, soft but defined — believable symmetry, not doll-like perfection |
| Lips | Full, natural rose-pink |
| Body | Slim, fit, elegant — not hypersexualized proportions |
| Skin texture | Visible pores, subtle warmth, natural unevenness |
| Distinguishing | Delicate gold jhumka earrings, occasional bindi in lifestyle content |

**Isolation Rule (post-LoRA):** Never describe face shape, eye color, hair, skin tone, or ethnicity in prompts. The LoRA encodes these permanently. Re-prompting anatomy causes "double-baking" — plastic, distorted faces.

---

## What to Prompt (post-LoRA)

Only describe: clothing, setting, pose, expression, lighting, mood. Nothing about physical appearance.

**Good:** `rooftop portrait at golden hour, silk saree in emerald green, leaning on railing, wind in hair`
**Bad:** `AnanyaAI, dark brown eyes, long wavy hair, rooftop portrait` ← triggers double-baking

---

## Realism Principles

- **Photography language:** `shot on Canon EOS R5`, `35mm f/1.8`, `natural window light`, `golden hour`, `editorial fashion photography`
- **Not:** `8k ultra HD masterpiece best quality hyperrealistic` — pushes toward synthetic renders
- **Expressions:** soft smile, neutral confidence, mid-laugh, candid glance — not frozen beauty pageant pose
- **Composition:** mirror shots, rooftop portraits, cafe candids, hotel corridor walks, balcony golden-hour

---

## Output QC Checklist

Run before saving any image:
- [ ] Face believable at 100% zoom
- [ ] Hands pass review — count fingers, check joints
- [ ] Outfit seams, straps, jewelry consistent across the image
- [ ] Background physically plausible — no melting, no impossible objects
- [ ] Clearly adult presentation — mature styling, not childlike
- [ ] No visual artifact that would make a viewer say "this is AI" at first glance
- [ ] Free-tier: safe to post publicly on Instagram
- [ ] Premium-tier: exclusive/glamorous but fully clothed and Instagram-compliant

---

## Content Tiers

| Tier | Platform | Output Folder | Notes |
|------|----------|--------------|-------|
| Free / Lifestyle | Instagram (public) | `output/YYYY-MM-DD/ananya/` | Aspirational fashion, lifestyle |
| Premium / Subscriber | Instagram Subscriptions | `output/YYYY-MM-DD/ananya/` | Editorial glam, body-confident |

All content remains **fully clothed and Instagram-compliant**. Final outputs require human review before posting.

---

## Disclosure Requirements

- Instagram: every post must include `#AI` in caption
- AI creator label must be visible on profile

---

## Training Dataset Specs

- Resolution: 1024×1024 PNG
- Split: 8 closeup + 8 medium + 8 fullbody = 24 images minimum (from 48+ bootstrap candidates)
- Captioning: Isolation Rule — describe setting/pose/lighting ONLY. Never physical features.
- Format: one `.txt` file per image, same filename

---

## LoRA Training Settings

| Setting | Value |
|---------|-------|
| Trigger word | `AnanyaAI` |
| Steps | 2000–2400 (stop at best-looking visual checkpoint) |
| Network dim | 32 |
| Optimizer | Prodigy |
| Dataset | 24 curated synthetic images (8+8+8) |

---

## Legal Notes

- Training data: synthetic only — no real-person likenesses
- Juggernaut XL license: verify commercial use terms before monetization
- Ananya is a fully synthetic persona — no real person is depicted
- Maintain records that training images are AI-generated (bootstrap_seeds.py output)
