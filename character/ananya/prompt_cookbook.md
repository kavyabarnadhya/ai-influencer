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

## Body — Updated Guidance (from cafe_morning_03 review)

- Current body tokens work for identity consistency but **hourglass + good busts not prominent enough** when dress is loose/relaxed fit
- For male-skewing content: use **fitted/bodycon silhouettes** — loose linen/relaxed dresses hide curves entirely
- Add `hourglass figure, defined waist, fuller bust` explicitly alongside existing M-size tokens — these reinforce each other
- Avoid: relaxed fit, oversized, flowy, A-line — all flatten the figure

---

## Outfit — Trending (male audience, bold/seductive-adjacent, Instagram-safe)

| Prompt | Status | Notes |
|--------|--------|-------|
| `emerald green fitted bodycon dress, scoop neck, sleeveless, above-knee hem, fitted through hips and bust` | ✅ recommend | Jewel tone, bodycon, sleeveless — safe + seductive |
| `electric blue satin slip dress, thin straps allowed, above-knee hem, fitted through hips` | test | Bold color, drapes body — high risk of strap drift, add `thin straps` to negative |
| `fiery scarlet fitted midi skirt, high waist, pencil silhouette, below-knee hem` + `scoop neck fitted crop top, sleeveless, scarlet` | ✅ recommend | Co-ord — shows waist strongly |
| `cobalt blue fitted crop top, square neck, sleeveless, thick straps` + `high-waisted wide-leg cobalt palazzo pants` | test | Bold co-ord, less seductive but trendy |
| `deep burgundy bodycon mini dress, square neck, short sleeves, above-knee hem` | ✅ recommend | Dark jewel tone + bodycon — strong |

**Rule:** For male audience carousels — **always bodycon or fitted, never relaxed/flowy**. Color bold (jewel tone or fiery), not neutral.

---

## Background — Sharp/Premium (from cafe_morning_03 learnings)

cafe_morning_03 issue: background drifted to street instead of cafe terrace. No terrace furniture visible in wide/medium slides.

**Fix:** Be explicit with visible props in scene prompt — name them: `marble cafe counter`, `rattan chairs`, `espresso machine`, `sunlit glass facade`.

| Scene | Prompt | Status |
|-------|--------|--------|
| Premium hotel lobby | `luxury hotel lobby, marble floors, tall glass windows, warm ambient lighting, premium interior, sharp background` | ✅ recommend |
| Rooftop pool bar | `rooftop infinity pool bar, Mumbai skyline background, golden hour, cocktail bar counter, sharp background` | ✅ recommend |
| Upscale restaurant terrace | `upscale restaurant terrace, rattan chairs, white tablecloth, candles, warm evening light, sharp background` | ✅ recommend |
| High-rise apartment | `floor-to-ceiling glass windows, city skyline view, luxury apartment interior, warm lamp light, sharp focus` | ✅ recommend |

**Always append:** `sharp background, environmental detail, realistic depth, f/8 aperture, deep focus, no lens blur` — already in `sharp_background_positive` config token. Confirm this is included in every slide prompt.

**low_bokeh_negative** already in config.yaml at `carousel.low_bokeh_negative` — verify it's injected in `build_negative()`.

---

## Outfits — Bodycon Cutout Problem

**Trigger:** `scoop neck` + bodycon combination → SDXL generates midriff cutout in ~60% of candidates even without prompting it.

**Fix:** Add to global negative: `midriff cutout, cutout dress, cutout detail, cut-out dress, side cutout, stomach cutout`

**Status:** Confirmed in emerald_06. All 3 wide candidates had cutout drift on some candidates. Cand 2 (wide+medium+close) was cleanest. Add these tokens to `config.yaml` negative_prompt before next run.

---

## Carousel Calibration Runs

| Run | Result | Notes |
|-----|--------|-------|
| `calibration_bodycon_dusk_v2_01` | pass with candidate selection | V2 generated 3 candidates per model role. Best selected set has distinct full-body, side/three-quarter medium, side close portrait, and ambient rooftop. Outfit color/hair/identity mostly held. |
| `carousel_cafe_morning_03` | partial pass — selected 4 slides | White linen relaxed dress: **too loose, hides figure**. Background drifted to street not terrace — no cafe props visible. Dress hem rendered mini not knee-length. Face identity strong across all candidates. cand_1 slide_2 + cand_1 slide_3: high dress slit = Instagram flag, rejected. Selected: slide_1 cand_2, slide_2 cand_2, slide_3 cand_3, ambient. **Learnings: use bodycon/fitted for male audience; scene needs explicit prop names; relaxed fit = no-go for this content goal.** |
| `carousel_hotel_lobby_emerald_06` | pass — best cand 2 across all slides | Emerald green bodycon + hotel lobby. Face identity **strong and consistent** — IPAdapter fix (0.5 weight, 0.15 start_at) working well. Img2img anchor fix on close slide working — no more face divergence between slides. **Issue: midriff cutout appeared on ~60% of candidates** (scoop neck + bodycon triggers this). Cand 2 across wide/medium/close was cleanest. Ambient: luxury hotel chandelier + city skyline — excellent. Settings used: medium denoise 0.70, close denoise 0.65, ControlNet close 0.60. |

V2 settings from this pass:

- `medium`: img2img denoise `0.74`, ControlNet `0.75`
- `close`: img2img denoise `0.82`, ControlNet `0.8`
- `carousel_production_v2/` pose bank gave meaningful pose/angle variation
- `slide_2_medium_cand_2_290916075.png` rejected after user review: double/deformed hand artifact. Its source pose was moved out of the active production set.
- Close candidates `1` and `2` drifted into off-shoulder styling; close candidate `3` kept the short-sleeve dress best
- Selected set saved in `output/2026-05-05/ananya/carousel_calibration_bodycon_dusk_v2_01/selected_best/`

Production pose guidance update: use `character/ananya/poses/carousel_production_v2/` for calibration and production until more poses pass review.
