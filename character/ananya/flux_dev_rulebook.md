# FLUX dev Prompt Rulebook — Ananya Pipeline
**Sources: Official HuggingFace model card, diffusers docs, fal.ai guide, skywork.ai guide, ComfyUI docs, empirical runs from this project.**

Stack: FLUX dev Q4_K_S GGUF + XLabs Realism LoRA 0.5, 20 steps, CFG 3.5, img2img denoise 0.80.

---

## RULE 1: Sentence Order (Verified — fal.ai + skywork.ai)

FLUX weighs earlier tokens more heavily. Burying the subject late = deprioritized output.

**Required order every prompt:**
```
1. Subject + body descriptor
2. Outfit (fabric + color + fit)
3. Pose / action
4. Location / environment
5. Lighting
6. Camera / lens
7. Skin + realism tail
```

Never: "beautiful sunset in Mumbai, wearing red saree, a woman with..."
Always: "23-year-old South Asian woman... wearing... standing at... with golden hour light..."

---

## RULE 2: Natural Prose, Not Tag Lists (Verified — fal.ai, skywork.ai)

FLUX responds to conversational, literal language. Dense keyword lists cause variable results.

❌ `woman, red dress, beach, sunset, bokeh, 8k`
✅ `A woman in a red silk dress standing barefoot on a sandy beach at sunset, warm golden light behind her, shallow depth of field`

Specificity = predictability. Camera model, fabric type, lighting direction — each one anchors output.

---

## RULE 3: Prompt Weights Don't Work (Verified — getimg.ai official)

`(token:1.3)` CLIP syntax silently ignored in FLUX. Use descriptive phrasing instead.

❌ `(hourglass:1.4), (fitted kurta:1.3)`
✅ `M-size hourglass figure with defined waist... wearing fitted tailored kurta cinched at waist`

---

## RULE 4: Negative Prompts — Weak at CFG 3.5 (Verified — empirical runs)

CFG 3.5 = negatives have near-zero effect. All avoidance must be positive phrasing.

❌ `no plastic skin, not slim, no flowing fabric`
✅ `visible skin pores, natural skin texture, M-size figure, fitted tailored fabric`

Official recommended CFG: 3.5 (HuggingFace model card). Going lower (1.5–2.5) improves skin realism but reduces prompt adherence — tradeoff.

---

## RULE 5: Body + Clothing Interaction (Empirically validated — this project, 2026-05-18)

**Fabric geometry overrides body tokens.** Loose fabric = FLUX renders volume = reads as larger body.

| Outfit type | What FLUX does | Fix |
|---|---|---|
| Loose ethnic (kurta, saree pallu, dupatta) | Fabric volume = plus-size rendering | `fitted/tailored/cinched` + specify which shoulder for dupatta |
| Form-fitting western (slip dress, bodycon) | Defaults to slim editorial | Add `soft natural curves, fabric shows soft midsection` |
| Layered (overshirt over tank) | Overshirt hides silhouette | `unbuttoned worn OPEN, sleeves rolled` + `fitted ribbed tank underneath` |
| Saree specifically | Pallu adds ~1 size visually | `fitted blouse showing waist definition` BEFORE saree token |

**Body MUST come before outfit in sentence order.**

---

## RULE 6: M-Size Hourglass — Use Redundant Tokens (Empirically validated)

Single token like "curvy" insufficient. Need 3-4 overlapping cues:

✅ Reliable stack:
```
M-size hourglass figure, soft natural curves not slim editorial, not plus-size, average Indian build
```

Note: `defined waist and full hips` was removed 2026-05-21 after A/B test showed simplified stack produces better results with Body FIX LoRA 0.7 — more natural proportions, less over-engineered.

❌ Fails:
- `curvy` alone → plus-size
- `thick` → fitness-magazine
- `voluptuous` → cleavage-forward
- `slim` + curve token → FLUX picks slim
- Weight numbers (60kg) → ignored

---

## RULE 7: Dupatta / Draped Fabric (Empirically validated — this project)

Failures: disappears, merges with clothing, renders symmetric.

**Required pattern:**
```
[CONTRASTING COLOR] [FABRIC] dupatta draped asymmetrically over LEFT shoulder,
loose end falling to right hip, visible fold lines and natural drape weight
```

Rules:
- Contrasting color to garment — same color = merger
- Name the shoulder (left/right) — generic "over shoulder" = random/confused
- Fabric weight cue: "soft chiffon", "starched cotton", "heavy silk"
- `asymmetric` prevents mirror-perfect drape AI tell

For saree pallu: `pallu pleated and pinned at left shoulder, falling behind to mid-calf`

---

## RULE 8: Location — Descriptive Scaffolding Required (Empirically validated)

Named neighborhoods alone produce generic results. Scaffold with physical description.

✅ Upscale:
```
in upscale South Delhi neighborhood, manicured trees lining wide clean pavement,
restored colonial bungalow, freshly painted walls, Lodhi Colony aesthetic
```

✅ Working named locations (FLUX renders distinctly):
- "Pondicherry French Quarter" → yellow walls, bougainvillea
- "Lodhi Art District" → pastel walls + murals
- "Jaipur old city" → pink sandstone walls
- "Goa village" → Portuguese arches, red tile

❌ "Hauz Khas Village" → unreliable. Use: `South Delhi village area near medieval ruins`

---

## RULE 9: Skin Realism — Verified Approach (Empirically validated + ThinkDiffusion)

**Working skin block:**
```
authentic skin texture with visible pores, natural skin imperfections,
realistic lighting and true color tones, no plasticky skin,
no overly smooth surfaces, organic human appearance,
shot on Sony A7IV 85mm, natural ambient light
```

CFG effect: lower CFG (1.5–2.5) naturally improves skin but trades prompt adherence.
At CFG 3.5: must explicitly front-load skin tokens.

**Camera model anchor** — verified working approach:
- `shot on Sony A7IV 85mm` → cinema-adjacent skin + bokeh
- `shot on Fujifilm X-T5` → film-grain skin + warm tones
- `shot on iPhone 14` → casual snapshot, breaks studio editorial bias

Note: `IMG_XXXX.HEIC` filename trick — **unverified**. Do not rely on it.

---

## RULE 10: Anti-AI-Tell Tokens (Verified — ThinkDiffusion + empirical)

```
asymmetric facial features, slightly off-center composition,
candid moment not posed, one side of face in soft shadow,
weight shifted on one hip not centered
```

Common AI tells: bilateral symmetry, dead-center composition, both feet flat, catalog pose, perfect jewelry.

---

## RULE 11: Indian Embroidery — Realistic Expectations

FLUX renders: floral motifs, paisley, whitework density, fabric sheen. Cannot distinguish stitch types.

Specify: **density + placement** (not stitch names).
```
white chikankari embroidery in floral motifs across yoke and hem,
dense at neckline, scattered on sleeves
```

Other textiles:
- Banarasi: `gold zari brocade, traditional buti motifs` ✓
- Block print: `indigo block print floral repeat` ✓
- Bandhani: `tie-dye dot pattern` ✓

---

## RULE 12: FLUX Dev Architecture Facts (Verified — HuggingFace official)

- Dual encoder: CLIP (`clip-vit-large-patch14`) + T5 (`t5-v1_1-xxl`)
- Max sequence length: **512 tokens** for dev (256 for schnell)
- Same prompt goes to both encoders unless `prompt_2` specified
- Recommended CFG: **3.5** (official)
- Recommended steps: **50** (official) — we use 20 for speed tradeoff
- 12B parameter rectified flow transformer

---

## RULE 13: img2img Denoise Practical Guidance (Empirically validated — this project)

| Goal | Denoise |
|---|---|
| Same outfit, pose variation only | 0.55–0.65 |
| Same outfit, color/texture change | 0.70–0.78 |
| New outfit, same body + location | 0.80–0.85 |
| New outfit + new location | 0.88–0.92 |

At 0.80: anchor pose/composition locks in, outfit changes render, body silhouette change minimal.
Body silhouette comes from anchor — write anchor prompt carefully, slides inherit it.

---

## MASTER TEMPLATE (Ananya, img2img, denoise 0.80)

```
23-year-old South Asian woman with M-size hourglass figure,
soft natural curves not slim editorial, not plus-size, average Indian build,
wearing [FITTED outfit + fabric + color + fit — use tailored/cinched/body-skimming],
[asymmetric pose, weight on one hip, candid not catalog stance],
in [named Indian location] with [3-4 physical environment descriptors],
[lighting direction + color temperature],
shot on [Sony A7IV / Fujifilm X-T5] [focal length],
authentic skin texture with visible pores, natural skin imperfections,
no plasticky skin, asymmetric facial features, slightly off-center composition,
one side of face in soft shadow, candid moment not posed
```

---

## ETHNIC WEAR TEMPLATE (kurta, saree)

```
23-year-old South Asian woman with M-size hourglass figure,
soft natural curves not slim editorial, not plus-size,
wearing FITTED [garment] TAILORED close to body with subtle cinch at waist,
[fabric + color + embroidery placement],
[CONTRASTING COLOR] [fabric weight] dupatta draped asymmetrically over LEFT shoulder,
loose end falling to right hip with visible fold lines,
[pose], [location + scaffolding], [lighting],
shot on Sony A7IV 50mm,
authentic skin texture with visible pores, asymmetric features, off-center composition
```

---

## WESTERN FORM-FITTING TEMPLATE (slip dress, bodycon)

```
23-year-old South Asian woman with M-size hourglass figure,
soft natural curves not slim editorial, fabric shows soft midsection,
not plus-size, average Indian build,
wearing FITTED [garment] that follows natural body curves,
[pose], [location + scaffolding], [lighting],
shot on Sony A7IV 85mm,
authentic skin texture with visible pores, asymmetric features, off-center composition
```

---

## LAYERED WESTERN TEMPLATE (overshirt, jacket)

```
23-year-old South Asian woman with M-size hourglass figure,
soft natural curves not slim editorial, not plus-size,
wearing high-waisted [bottom] accentuating waist,
fitted [inner top], unbuttoned [outer layer] worn OPEN with sleeves rolled,
[pose], [location + scaffolding], [lighting],
shot on iPhone 14 [focal length],
authentic skin texture with visible pores, asymmetric features, candid snapshot
```

---

## RULE 14: Neckline Contradictions (Empirically validated — 2026-05-20)

**Never mix neckline descriptors that contradict each other.** FLUX picks one and ignores the other unpredictably.

❌ `halter-neck bodycon dress with straight bandeau neckline` → FLUX rendered strapless bandeau, ignored halter
✅ `halter-neck dress with neck strap tied behind neck, bare shoulders, open back`

**Rule:** One neckline descriptor only. Make it physical and specific:
- Halter: `neck strap tied behind neck, bare shoulders, open back`
- Bandeau/strapless: `strapless straight neckline, no straps, bare shoulders`
- Off-shoulder: `elasticated off-shoulder neckline sitting below collarbone`
- V-neck: `deep V-neckline`

**For standing slides with CN ≥ 0.70:** ControlNet pose lock at high strength can override neckline from shared_tail. Always repeat neckline descriptor explicitly in each standing slide prompt.

---


## RULE 15: Kontext OOTD Carousel — Validated Recipe (2026-05-20)

**Confirmed working stack for multi-slide outfit-locked carousels:**

### Workflow config
- Model: `flux1-dev-Q4_K_S.gguf`
- Realism LoRA: `flux-xlabs-realism.safetensors` strength **0.5** (node 15)
- Body LoRA: `Body FIX FLUX.safetensors` strength **0.7** for bodycon/fitted dresses (node 17)
  - Set via YAML `anchor_body_lora_strength: 0.7`
  - Use `0.0` (disabled) if dress is loose/flowy — LoRA amplifies fabric volume
- BG lock: **automatically appended** to every Kontext slide via `_inject_flux_kontext()`:
  `, same background, same scene, unchanged environment`
  — no per-file changes needed, applies universally

### Anchor prompt structure
```
[body descriptor 4-token stack]
wearing [outfit — physical neckline + fabric + fit]
[asymmetric starting pose, weight on one hip]
on [location — physical descriptors matching slide prompts exactly]
[lighting]
shot on Sony A7IV 50mm
[skin realism tail]
```

### Slide prompt structure
```
anchor=standing | same woman in same [brief outfit name] [new pose/framing],
[optional: single accessory callout], [optional: expression],
keeping exact same [neckline geometry], [left/right shoulder detail], [fabric detail], and [accessories]
```

### BG consistency rule
**Slide prompts must describe the same scene as the anchor — no scene invention.**
Bodyfix_test failed because slides said "rooftop + golden hour" while anchor was "apartment balcony + daylight."
v1 and red dress carousel passed because slides matched anchor scene tokens.

### Per-outfit body LoRA strength
| Outfit type | `anchor_body_lora_strength` |
|---|---|
| Bodycon midi / fitted dress | `0.7` |
| Midriff / crop top | `0.4` |
| Loose / flowy / ethnic | `0.0` |

### Validated results
- `carousel_black_oneshoulder_ruched_v1` — no body LoRA (pre-recipe), BG consistent, natural slim
- `carousel_black_oneshoulder_bglock_test` — Body FIX 0.7 + BG lock, curvy + consistent ✓
- `carousel_red_oneshoulder_ruched_v1` — full 6-slide validation, all 3 targets met ✓ **(gold standard)**

---

## WHAT TO IGNORE (unverified claims stripped from prior rulebook)

- "40% influence loss past 3rd clause" — fabricated specific number
- IMG_XXXX.HEIC filename trick — no verified source
- XLabs LoRA slim bias — undocumented, not on model card
- Denoise "below 0.95 = SD 0.50" — unverified specific claim
- "fabric shows soft midsection" as documented token — invented; works empirically but is our own heuristic
