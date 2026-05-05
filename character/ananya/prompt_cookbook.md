# Ananya Prompt Cookbook

Reference before writing any carousel or batch prompts. Updated after each reviewed run.

---

## Body

| Token | Result | Notes |
|-------|--------|-------|
| `realistic size M Indian woman, medium curvy build, fuller hips and thighs, natural waist, soft stomach, fuller bust, realistic proportions` | use going forward | Stronger positive body descriptor. Replaces `not skinny` phrasing, which SDXL ignores or misreads. |
| `size M body, medium build, fuller bust, soft stomach, realistic curves, not skinny` | deprecated | Partially worked, but `not skinny` is weak/misread. Use the stronger positive M-size descriptor above. |
| `medium build, hourglass figure, soft curves, realistic proportions` | ❌ Too weak | LoRA ignores it, renders slim |

---

## Outfits — Tops

| Prompt | Result | Notes |
|--------|--------|-------|
| `V-neck teal shirt, long sleeves, loose fit` | ✅ Reliable | Clear silhouette, SDXL renders consistently |
| `deep teal wrap top` | ❌ Drifts | Renders as crop top or button-down across slides |
| `oversized cream linen shirt, rolled sleeves` | ❌ Drifts | Renders as short-sleeve top + shorts |
| `V-neck fitted crop top, scarlet red, V-neck no straps` | ❌ Drifts to spaghetti strap bra-top | "no straps" in positive prompt ignored — SDXL defaults to bra-top silhouette. Fixed by adding spaghetti straps to global negative. |
| `scoop neck fitted crop top, sleeveless, thick shoulder straps` | test after negative fix | — |

---

## Outfits — Dresses

| Prompt | Result | Notes |
|--------|--------|-------|
| `white linen dress, spaghetti straps, relaxed fit` | ✅ Consistent across 3 slides | Outfit held in cafe_morning_02 — spaghetti strap visible in all slides |

| `bodycon midi dress, scoop neck, short sleeves, berry pink, fitted through hips and thighs, knee-length hem` | partial pass | Color, bodycon silhouette, sleeves, hair, and identity held across 3 model slides with img2img+ControlNet. Hem rendered mini/above-thigh instead of knee-length midi. |

---

## Outfits — Tips

- Avoid: wrap, draped, asymmetric — SDXL resolves ambiguity incorrectly
- Prefer: describe neckline + sleeve length + fit explicitly
- Add `long hem past hips` for shirts/tops to prevent shorts rendering
- Simpler = more consistent: one garment name + 2 descriptors max

---

## Hair

| Prompt | Result | Notes |
|--------|--------|-------|
| `loose waves, side parted, natural` | ✅ | Renders well, natural movement |
| `messy bun, casual` | ✅ | Consistent across slides |
| `sleek straight hair, centre parted` | ❌ LoRA overrides | Renders as loose waves — LoRA encodes wavy/natural hair strongly, ignores straight descriptor |

---

## Jewelry

Jewelry is randomised from pool in code — no need to prompt it manually.
Pool already tuned: small/subtle gold/silver pieces only.

---

## Face

Never re-prompt face features (eyes, skin, ethnicity) — LoRA encodes these.
Only prompt mood/expression via pose role tokens (already in SLIDE_ROLES).

---

## Scenes / Environments

| Prompt | Result | Notes |
|--------|--------|-------|
| `cafe terrace, soft morning light, warm golden hour, Mumbai street view below` | ✅ | Clean Mumbai street cafe, golden hour haze |
| `cozy indoor cafe, warm amber lamp light, wooden table, book and coffee cup` | ✅ | Warm amber interior, props render well |

---

## Ambient Slides (no model)

Ambient slides use `t2i_sdxl_lora.json` — no IPAdapter/ControlNet.
Scene prompt alone drives the image. Results have been strong.

| Scene | Ambient result |
|-------|---------------|
| cafe terrace, golden hour, Mumbai street view | Rooftop cafe terrace, city skyline, hazy golden light ✅ |
| cozy indoor cafe, warm amber lamp, book and coffee | Amber lamp + book + coffee on wooden table ✅ |

---

## Known Issues

- **Production pose set**: Use `character/ananya/poses/carousel_production_v2/` for calibration and production until more poses pass review. It uses role-prefixed candidate poses and avoids the random full pose pool.
- **Workflow consistency fix**: Model slides after slide 1 should use img2img anchor + ControlNet together. This held berry-pink bodycon color and silhouette across calibration slides much better than independent t2i.

- **Slide 1 (wide) framing too tight**: Some standing skeletons in `carousel_varied/` crop to medium-close. If wide shot needed, use `carousel_default/` or a specific `--poses-dir` with only true full-body skeletons.
- **Outfit drift on close slide**: Happens when bust skeleton (close_01–08) conflicts with garment shape. Simpler top descriptions reduce this.
- **Outfit drift on medium slide (3/4 angle)**: Different pose angle causes SDXL to re-resolve garment. Mitigation: add explicit color lock (`scarlet red, bold red color`) + neckline exclusion (`V-neck, no straps`) to prevent spaghetti strap render.
- **Color drift**: Top color can shift (scarlet → pink) across slides. Always add color adjective twice: garment name + `bold [color]` token.
- **Spaghetti strap default**: SDXL strongly defaults to spaghetti strap / bra-top for crop tops. Fixed globally — `spaghetti straps, thin straps, bra top, bandeau top, tube top, strapless top` added to config.yaml negative_prompt. Do not attempt to fix per-prompt.
- **Negative descriptors in positive prompt fail**: `"no straps"`, `"not skinny"` style tokens are ignored or misread. Use global negative for structural exclusions.
- **Body realism**: If she reads too lean, use positive M-size body tokens (`medium curvy build`, `fuller hips and thighs`, `soft stomach`) and keep slim-body terms in the global negative prompt. Avoid relying on `not skinny`.

---

## Carousel Calibration Runs

| Run | Result | Notes |
|-----|--------|-------|
| `calibration_bodycon_dusk_v2_01` | pass with candidate selection | V2 generated 3 candidates per model role. Best selected set has distinct full-body, side/three-quarter medium, side close portrait, and ambient rooftop. Outfit color/hair/identity mostly held. |

V2 settings from this pass:

- `medium`: img2img denoise `0.74`, ControlNet `0.75`
- `close`: img2img denoise `0.82`, ControlNet `0.8`
- `carousel_production_v2/` pose bank gave meaningful pose/angle variation
- `slide_2_medium_cand_2_290916075.png` rejected after user review: double/deformed hand artifact. Its source pose was moved out of the active production set.
- Close candidates `1` and `2` drifted into off-shoulder styling; close candidate `3` kept the short-sleeve dress best
- Selected set saved in `output/2026-05-05/ananya/carousel_calibration_bodycon_dusk_v2_01/selected_best/`

Production pose guidance update: use `character/ananya/poses/carousel_production_v2/` for calibration and production until more poses pass review.
