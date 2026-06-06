# Ananya Carousel Workflow — Canonical Reference

**Last updated: 2026-06-06 — NEW §14 ultra-realism pass (`--ultra` selective realism refiner) + §15 `lens_profile:` key (editorial bokeh vs selfie deep-focus). Validated white bodycon club + black tube daycafe carousels: ultra adds real skin pore / hair / fabric micro-texture BEFORE ReActor (face identity guaranteed), background untouched. Uplift most visible in bright daytime deep-focus scenes; moderate in dark neon. Sweet-spot denoise 0.38; per-slide `ultra=0.44` for hero closeups; >0.50 over-processes fabric.**

Prior 2026-06-03 — §8 new forbidden patterns from black cowl NYE v1: object duplication on close detail shots (force "ONE single"); `faceswap=false` does NOT guarantee faceless (crop head out of frame); open-back drifts to racerback on back/walk-away views (re-specify single nape tie); side-profile bag-arm bends backward (pose arm forward natural).

Prior 2026-05-31 — §7 faceless/off-camera slide vocabulary + `faceswap=false` token (skip ReActor for zero-face slides); §8 new forbidden patterns (mirror BG → portal artefact, hair-flip → rubbery hair, straight-overhead arms); §2 hair-color lock (warm scenes drift lighter) + skin caps lowered to 10/8 (25/12 over-bleached warm scenes). Validated red halter vanity v1.

Prior 2026-05-30 — §8 + §11: hands resting on a white/light skirt fuse every time (low contrast → YOLO hand-detect misses → Stage 3.5 never fires; `behind back` does not fix it). Only reliable fix = both hands above the bust on skin/hair/darker drape + regenerate the slide. Validated white off-shoulder lace corset v1.

Prior 2026-05-29 — skin lock Stage 3.6 now applies LAB shift to BOTH face and body skin (no face exclusion), `_MAX_L_SHIFT` raised back to 25.0 / `_MAX_AB_SHIFT` to 12.0. Warm-cast scenes (festive red, golden hour, indoor incandescent) were leaving the rendered face darker than face_ref_v2 baseline via ReActor blend bleed-through; lifting face+body together unifies the whole subject to the fair reference tone (validated on red chikankari lehenga festive v9d). `scripts/comfyui_api.py` discovery sweep now includes port 8001 (ComfyUI Desktop fallback when 8000 is held by a zombie process). §11 hand realism criterion gains an explicit third-hand check (slide_03 of v9d had hair-touch+chest+hip = 3 hands; partial rerun with strict 2-hand prompt fixed it).

Prior 2026-05-27 — §11 review checklist adds hand realism criterion (Stage 3.5 occasionally misses on cup-grip / on-railing / fingers-spread poses); §11 documented exception section adds failure-resolution modes (drop-slide ship 5, reroll post-process, partial rerun) with guidance on when to pick which. Validated 2026-05-27 on black puff crop haveli v1 (slide_00 hand deformity → dropped, shipped 5-slide carousel).

Prior 2026-05-26 — body push formula now fabric-aware (§2: slimming fabrics like leather need a softer push; voluminous fabrics need none). §5 BG rules add floor cleanliness override + event-installation wall-floor colour match. Added black strapless leather maxi event v5 worked example (§12). Validated skin lock feather patch in production (orange tank cafe v3 + black leather event v5, ΔL up to 13.7 with no burn).

Prior 2026-05-25: skin lock Stage 3.6 Gaussian-feathers body skin mask (σ=8px) before LAB shift, eliminates seam halo when ΔL > 10. Added `scripts/reprocess_carousel_post.py` (deterministic hand seed, `--seed-base N` to reroll). Body push prompt formula, slide-vs-anchor vocab lock rule, ribbed bodysuit + wrap skirt outfit-type rules, reprocess-script subsection.**

This is the **single source of truth** for generating any Ananya OOTD/lifestyle carousel. Read this end-to-end before starting a new carousel. CLAUDE.md contains the short pre-flight gate; this doc contains the full reference and worked examples.

---

## 1. Pre-flight checklist (run through every time)

Before typing any carousel command:

1. **Read this file in full.** Not skim — read. A pointer that says "always read X" is worthless if X is never opened.
2. **Confirm anchor YAML has identity locks:**
   - `face_ref: character/ananya/seeds_v2/face_ref_v2.png`
   - `anchor_seed: 334521876`
   - `anchor_body_lora_strength: 0.5` (or `0.0` for headshot / flowy linen exceptions)
3. **Confirm the YAML header says `# Used with: --flux-dev --kontext flags (MANDATORY)`.** If missing, add it before running.
4. **Confirm your command includes `--flux-dev --kontext`.** Without these flags, the pipeline silently downgrades to FLUX img2img which locks anchor composition → all 6 slides become same pose.
5. **Always run `--anchor-only` first.** Wait for user approval. Then run full carousel. Never skip the gate — a broken anchor wastes ~12 min of GPU on 6 bad slides.
6. **Read your slide prompt file end-to-end** and verify none of the forbidden patterns from Section 8 appear.

If any step fails, fix it before proceeding.

---

## 2. Identity locks (immutable foundations)

These never change across any Ananya carousel. They are the foundation of cross-slide consistency.

| Lock | Value | Where set | Why |
|---|---|---|---|
| Face ref | `character/ananya/seeds_v2/face_ref_v2.png` | `face_ref:` in anchor YAML | ReActor swaps this onto every slide post-FLUX. Body skin lock matches body tone to this face's cheek LAB. |
| Body seed | `334521876` | `anchor_seed:` in anchor YAML | Validated M-size hourglass. Derives skin tone naturally with `South Asian woman` token. |
| Body LoRA strength | `0.5` universal | `anchor_body_lora_strength:` in anchor YAML | Exceptions: `0.0` for headshot (no body) or flowy/loose linen (LoRA amplifies fabric volume). |
| Realism LoRA | `0.5` baked into workflow node 15 | `workflows/flux_dev.json` | Do not change. |
| Hand realism post-process | Auto-applied | `workflows/flux_hand_detail.json` | Runs as Stage 3.5 after ReActor — SDXL FaceDetailer inpaint on `hand_yolov8s.pt` bbox using Juggernaut XL. Fixes FLUX 6-finger / deformed-hand artefacts (especially cup-grip poses). Sweet-spot tuned: denoise 0.50, cycle 1, bbox_dilation 15, feather 8. Higher denoise (0.65+) makes hands look heavy/uncanny. ~10-15s/slide. ORDER: must run BEFORE skin lock — hand inpaint may shift hand skin tone; subsequent skin lock unifies whole body to face_ref. |
| Skin tone post-process | Auto-applied | `scripts/skin_color_match.py` | Runs after hand detail. Locks body skin to face_ref cheek LAB. Mask Gaussian-feathered σ=8px before LAB shift — required to avoid seam halo at body silhouette when ΔL > 10 (validated on orange tank cafe v3). LAB shift caps `_MAX_L_SHIFT=10`, `_MAX_AB_SHIFT=8` (lowered 2026-05-31): the old 25/12 over-corrected warm-tungsten indoor scenes to a plastic foundation-pale look. No prompt action needed. |
| Hair color | `long dark brown hair` in anchor + every slide vocab lock | anchor `anchor_prompt` / `shared_tail` + slide `keeping exact same` | Warm-tungsten / amber-sconce scenes drift hair lighter (renders auburn/caramel/blonde). Lock explicitly: `long dark brown hair NOT auburn NOT blonde NOT highlighted`. Validated red halter vanity v1 (2026-05-31). |

### Hard NEVER rules

- **NEVER add explicit skin tone tokens** (`warm medium-brown`, `tan complexion`, `olive`, `medium South Asian skin`). The seed + `South Asian woman` derives the correct tone. Adding tokens overrides the seed → wrong/darker tone.
- **NEVER re-prompt anatomy** (face shape, eye color, hair color, ethnicity, age beyond `23-year-old`). The seed + ReActor handle these. Re-prompting causes "double-baking" → distorted face.
- **NEVER swap the body seed** mid-project. `334521876` is the validated reference. Other seeds (`837492016`, `112847593`) produced wrong body types and are blocklisted.

### Body push prompt formula (when FLUX drifts slim editorial)

Seed 334521876 + body LoRA 0.5 + `South Asian woman with M-size hourglass figure` is the baseline. On form-fitting bodysuits, ribbed tanks, and tight-bodice outfits FLUX still drifts slim/editorial. When that happens, add to the anchor_prompt:

> `size 12 curvy body with full M-size hourglass figure, fuller heavier curvier build with soft visible belly midsection, wide hips noticeably wider than waist, thick fuller thighs touching, fuller décolletage, defined cinched waist with curves above and below, distinctly NOT slim NOT editorial NOT model-thin, average everyday curvy Indian woman build`

Key tokens that landed it (validated orange tank cafe v3, 2026-05-25): concrete size (`size 12`), negative comparisons (`NOT slim NOT editorial NOT model-thin`), and body-part-specific cues (`wide hips noticeably wider than waist`, `thick fuller thighs touching`, `soft visible belly midsection`). Weaker phrasing (`fuller curvier body`, `soft natural curves`) alone is not enough — FLUX bias is strong.

**Fabric-aware calibration — use a softer push on slimming fabrics.** The full body push above is calibrated for light/airy fabrics (cotton ribbed tanks, mini frocks) where FLUX defaults to slim editorial. **On slimming fabrics (black leather, dark sheath, structured bodice) the full push over-corrects to plus-size.** Validated on black strapless leather maxi event v5 (2026-05-26): the full push produced plus-size; the slim-default push produced slim editorial; the balanced middle landed M-size hourglass:

> `M-size hourglass figure, pronounced hourglass with hips noticeably wider than waist, fuller chest and décolletage, defined cinched narrow waist with soft curves at hips and bust, healthy curvy M-size Indian woman build, NOT slim editorial NOT model-thin NOT plus-size NOT oversized, balanced M-size proportions like a real event-goer with natural feminine curves not exaggerated`

Key difference: drop `size 12`, drop `thick fuller thighs touching`, drop `fuller heavier curvier build`. Keep `hips noticeably wider than waist` + `fuller chest`. Add `NOT plus-size NOT oversized` as a brake. Add `not exaggerated` qualifier.

| Fabric / outfit weight | Push to use |
|---|---|
| Light/airy: cotton tank, ribbed bodysuit, mini frock, chiffon slip | Full push (size 12 + fuller heavier + thick thighs) |
| Slimming: leather, dark sheath, structured pencil maxi, tight knit | Balanced push (drop size 12 + fuller heavier; keep hips>waist + fuller chest; add NOT plus-size NOT oversized) |
| Voluminous: lehenga, A-line, layered, draped | Minimal push (baseline `M-size hourglass with defined waist` may be enough; the fabric carries the visual volume) |

---

## 3. Anchor-first approval gate (mandatory two-stage workflow)

Every new carousel runs in two stages. Stage A must pass before Stage B is invoked.

### Stage A — Anchor only

```powershell
python scripts/faceswap_carousel.py `
  --anchor-config character/ananya/anchor_libraries/<name>.yaml `
  --name <name>_v1 --flux-dev --kontext --anchor-only
```

Output: a single `anchor.png` in `output/YYYY-MM-DD/ananya/<name>_v1/`. Roughly 2-3 minutes on RTX 3050 6GB.

**User reviews:** outfit correct? BG correct? Body proportions M-size? Face roughly Ananya (will be faceswapped later, but anchor face influences body skin tone)?

- If yes → proceed to Stage B.
- If no → iterate on the anchor YAML `anchor_prompt:` field, rerun Stage A. Do not advance to Stage B with a broken anchor.

### Stage B — Full carousel

```powershell
python scripts/faceswap_carousel.py `
  --anchor-config character/ananya/anchor_libraries/<name>.yaml `
  --prompts character/ananya/carousel_prompts/<name>.txt `
  --name <name>_v1 --flux-dev --kontext
```

Reuses the anchor from Stage A. Generates 6 Kontext edits + ReActor faceswap + skin lock per slide. Roughly 10-15 minutes on RTX 3050 6GB.

**Why two stages:** A broken anchor (wrong outfit, wrong BG, wrong body) propagates to all 6 slides. Catching it after Stage A saves ~12 minutes and ~6 candidates of wasted GPU time.

---

## 4. Outfit consistency rules

The anchor describes the outfit physically (geometry, fabric, color) **once** in the YAML. Each slide prompt repeats the `keeping exact same [details]` block to anchor Kontext on the outfit.

- **Describe by physical geometry, not garment name.**
  - ❌ `halter-neck dress` → ambiguous, Kontext invents variations
  - ✅ `neck strap tied behind neck, bare shoulders, open back` → physical and specific
- **No contradictory neckline descriptors in one prompt.** Pick one neckline (halter / bandeau / V-neck / off-shoulder) and stick to it. See Rule 14 in `flux_dev_rulebook.md`.
- **Fabric weight cue required** for any draped element: `soft chiffon`, `starched cotton`, `heavy silk`, `crinkle viscose`.
- **Every slide's `keeping exact same` block** should repeat at minimum: neckline geometry, strap/sleeve detail, waist detail, hem detail.
- **Slide-prompt vocabulary MUST match anchor YAML vocabulary.** If anchor says `wrap mini skirt with smooth front panel and no shorts underneath` and slides say `mini skort`, Kontext will drift toward the slide's vocabulary on regen → outfit changes mid-carousel. Use the same garment noun and the same defining clauses in both files. Validated fail: orange tank cafe v3 — anchor said wrap skirt, slides said `skort` → first render came back as cuffed shorts with lace-up on both sides.

### Per-outfit-type rules

| Outfit type | Key prompt rules |
|---|---|
| Ethnic (kurta, saree, lehenga) | `fitted/tailored/cinched` — NEVER `flowing/loose`. Dupatta needs contrasting color + named shoulder. Pallu pinned at left shoulder + falling behind. |
| Form-fitting western (slip, bodycon, mini frock) | Add `fabric shows soft midsection` to avoid slim-editorial default. |
| Ribbed bodysuit / fitted tank | Use `very thin string-like delicate spaghetti shoulder straps thin as cords` if straps are meant to read as spaghetti. Default `thin spaghetti straps` reads as full ~1cm webbing. |
| Wrap mini skirt with side tie | `wrap mini skirt with smooth solid front skirt panel falling unbroken to mid-thigh and NO shorts underneath, single decorative ribbon lace-up corset-tie detail only on left hip side seam`. Without `NO shorts underneath` + `single ... only on left hip` the render comes back as cuffed shorts with lace-up on both sides. |
| Layered western (overshirt, jacket) | `unbuttoned worn OPEN, sleeves rolled` + `fitted ribbed tank underneath`. |
| Corset/bodice mini | Repeat `structured fitted bodice, thick wide shoulder straps` if straps are visual hero. Thin straps drift if not enforced. |

---

## 5. BG consistency rules

- **Slide prompts must describe the same scene tokens as the anchor.** No scene invention per slide.
- **BG lock token** `, same background, same scene, unchanged environment` is **auto-appended** by `_inject_flux_kontext()` in `scripts/faceswap_carousel.py`. No per-file action needed.
- Each slide prompt should name 3-4 BG physical descriptors so Kontext has anchor points to preserve. Example for pink café:
  > `ornate white iron arch behind, colorful mosaic tile column on left, white iron café chairs on right, lush green foliage`
- **Failure mode:** if a slide prompt says "rooftop golden hour" while the anchor was "balcony daylight" → BG diverges. Both must match.
- **Indoor BG quirk:** indoor flat light renders body skin darker. The skin lock corrects this post-process, no prompt fix needed (validated on sequin mesh ΔL=+13.1).
- **Floor cleanliness:** FLUX defaults "polished floor" / "marble floor" / "concrete floor" to scuffed-and-stained textures. Add explicit `clean spotless ... no stains no scuffs no dirt no marks no tile grout lines` to enforce a clean surface. Validated on black leather maxi event v5 (2026-05-26): bare `polished concrete or marble floor` produced visible dirt; `clean spotless ... no stains no scuffs no dirt no marks` rendered a clean glossy surface.
- **Event-installation BG (wall-floor color match):** for store launches, brand activations, step-and-repeats where the floor is painted/carpeted to match the wall, use `clean polished <colour> floor underfoot painted same <colour> as the wall behind so wall and floor blend into one continuous uniform <colour> event installation surface`. Without this, FLUX renders a contrasting grey concrete/marble floor that breaks the immersive backdrop. Validated on black leather maxi event v5 with `terracotta-red wall + terracotta-red floor`.

---

## 6. Accessories consistency rules

- Repeat accessories in every slide's `keeping exact same` block: `gold cuff bracelet on left wrist, small gold hoops, gold ring on right hand`.
- **Kontext does NOT guarantee pixel-identical accessories.** Expect drop-in/drop-out across slides. Acceptable as long as core outfit (neckline, fit, color) holds.
- For accessories that must read clearly (brand placement, statement piece): use a closeup slide (slide_01 or slide_05) framed to feature them.
- **Footwear drifts more than upper-body accessories.** Don't expect identical shoes across slides — describe footwear once in the anchor and accept variations.

---

## 7. Standard 6-slide order + pose vocabulary

The benchmark order. Every new carousel should follow this unless there's a documented reason not to.

| Slot | Purpose | Pose vocabulary (pick one or compose) |
|---|---|---|
| slide_00 | Full body scroll-stop hook | `right hand on hip, left arm relaxed`, `mid-stride walking toward camera`, `weight shifted hard onto right hip with confident gaze` |
| slide_01 | Chest-up portrait closeup | `change to chest-up portrait framing showing face neck shoulders and neckline only, direct camera gaze` |
| slide_02 | Full body dynamic | `mid-stride walking`, `leaning forearm on railing`, `right hand raised pushing wavy hair off face` |
| slide_03 | Full body alt angle | `three-quarter angle facing slightly left/right TOWARD camera`, `leaning shoulder against pillar`, `body angled with head turned to camera` |
| slide_04 | Full body candid | `both hands relaxed at sides looking slightly off-camera`, `weight on left hip with soft smile`, `glancing to the right with candid expression` |
| slide_05 | Chest-up portrait closing | `chin slightly lowered with confident smirk`, `hand to cheek with fingers near jawline`, `looking up at camera with warm smile` |

### Pose rules

- **Each slide must have a uniquely identifiable composition.** Two slides with the same pose = failure (the user will spot it).
- **Closeups must use** the exact framing instruction: `change to chest-up portrait framing showing face neck shoulders and neckline only`. Variants like "waist-up" do NOT work — Kontext stays full-body.
- **All standing**, no mixing sitting in the same carousel. Sitting = separate post.
- **Hand-zone diversity:** vary hand placement across slides (at sides / in hair / at collarbone / one arm wide / raised overhead). Repeating "hand at collar/chest" on 4+ slides reads monotone. For an arms-raised/armpit slide, bake the raised pose into a **separate anchor group** (`armsup:`) — Kontext will NOT raise arms from a relaxed-arm anchor.
- Arms-raised should read sensual/languid (soft bent elbows, head tilt, heavy-lidded), NOT straight-overhead (reads as "police surrender").

### Faceless / off-camera slides (validated 2026-05-31, red halter vanity)

Candid faceless shots add editorial variety. Research-backed poses that work in Kontext:

| Faceless pose | Prompt approach | faceswap |
|---|---|---|
| Walking away toward a feature (sconce/window) | `from behind walking slowly away toward the [light], mid-stride, face NOT visible` | `faceswap=false` |
| Side profile holding a prop (bouquet) | `strict side profile facing left, both arms cradling a bouquet in front, head tilted up, face in profile not at camera` | keep ON (side face visible) |
| Bare back over-shoulder | `body turned three-quarter (~120°) away, bare back visible, head over shoulder` | keep ON (side face) |
| Looking down, hair forward | `chin to chest, hair falling forward over face, eyes NOT visible` | keep ON (side face) |
| Cropped torso / back of head | `cropped at collarbone, NO face in frame` or `180° back, only hair + bare back` | `faceswap=false` |

**Rule — faceswap toggle:** add `faceswap=false |` token (slide line) ONLY when **zero** Ananya face is visible (back of head, walking away, cropped torso). Skips ReActor → avoids it distorting hands/body near a non-existent face; hand detail + skin lock still run. If ANY side of the face shows (profile, looking-down), KEEP faceswap on — else FLUX's base (non-Ananya) face ships.

**Avoid:** the **hair-flip-across-face** pose — FLUX renders the flung hair artificially (rubbery strands, gaps showing face). Use walking-away or bouquet-profile for hidden-face instead.

---

## 8. Forbidden Kontext slide prompt patterns

| Pattern | Why it breaks | Replacement |
|---|---|---|
| `body turned away from camera` / `back to camera` | Full 180° flip → Kontext repaints scene geometry → BG collapses, invents new scene | `three-quarter angle facing slightly left/right toward camera, head turned to camera`. **Exception:** an intentional faceless slide (back of head / walking away) is fine — pair with `faceswap=false` and accept the BG may shift slightly; see §7 faceless table. |
| `hand touching ribbon tie` / `fingers on lace` / `hand on button/zipper/strings` | Kontext reads as untying/opening the garment | `hand to cheek with fingers near jawline`, `fingertips lightly at collarbone`, `hand raised near shoulder` |
| `sitting` in a standing carousel | Structural pose change → BG + outfit drift; Kontext fights its own preservation logic | Separate carousel entirely; sitting = standalone post |
| `change to waist-up portrait framing` | Kontext ignores waist-up cue, stays full-body | `change to chest-up portrait framing showing face neck shoulders and neckline only` |
| `mid-stride walking toward camera` / any motion verb from a static anchor | Kontext preserves anchor composition — motion instruction silently ignored → slide becomes near-duplicate of anchor pose | Use static-friendly action: `stepping forward with one foot ahead`, `cup raised to lips taking a sip`, `head tilted to one side looking off-camera` |
| `hand raised through/pushing hair` combined with `holding object in other hand` | Three-hand artefact — Kontext renders the raised arm but also keeps an "expected" hand at the side, producing extra limb | Pick one: either hair-push (both hands free from objects) OR holding object (no hair-push). If both poses needed, split across two slides. |
| `hand at hip` / `hand on skirt` / `hands holding skirt` on a **white or light-coloured skirt** | Low contrast — FLUX fuses the fingers into the pale fabric (clawed/webbed/fused hand). YOLO hand-detect (`hand_yolov8s.pt`) finds no clear hand bbox against white, so Stage 3.5 hand inpaint never fires → ships broken. `behind back` does NOT fix it: Kontext ignores the instruction and keeps a hand on the skirt anyway. | Keep BOTH hands **above the bust** on a high-contrast surface: fingertips at cheek/jaw, hand at collarbone (skin), hand in dark hair, or flat against a darker drape. Explicit tokens: `both hands held high above the bust, NO hand below the bust, NO hand on the skirt, NO hand on the hips`. Validated white off-shoulder lace corset v1 (2026-05-30) after 3 failed reruns — only hands-above-bust landed clean. |
| `mirror` / `gold-framed mirror` / any reflective surface in the BG | Kontext treats the mirror as a portal — renders the figure half-emerging from inside the mirror frame, or a garbled reflection. Even as a static anchor element it corrupts. | Use a non-reflective BG: `smooth cream-white textured wall`, `wall sconce`, `decorative wall panel`. No mirrors, ever (despite mirror-selfie being the common IG reference look). Validated red halter vanity v1 (2026-05-31). |
| `hair flip` / `hair flung across face` / hair in motion hiding face | FLUX renders flung hair as rubbery artificial strands with gaps that still show the face → looks fake. | For hidden-face use walking-away or side-profile-with-prop (see §7 faceless table). |
| `both arms raised straight overhead` (from a relaxed anchor) | Kontext won't raise arms from a relaxed-arm anchor (keeps anchor pose); when forced via a baked anchor, straight-up reads stiff/"surrender". | Bake an `armsup:` anchor group with a **languid sensual** stretch (soft bent elbows, head tilt). |
| `close detail shot of [handbag/object]` | FLUX duplicates the object — renders TWO handbags. Also `faceswap=false` does NOT guarantee faceless: FLUX still paints a (non-Ananya) face into the frame, which then ships unswapped. | Force singular: `ONE SINGLE bag, exactly one, NOT two NOT duplicate`. For truly faceless, crop the head OUT: `framed from the collarbone DOWN, entire head and face OUT of frame above, NO face NO chin visible` — don't rely on `faceswap=false` alone. Validated black cowl NYE v1 (2026-06-03). |
| open-back / halter top on a **back-view or walking-away** slide | Kontext re-invents the back as a `racerback` / `scoop-back` / crossed-straps — loses the open-back design. | Re-specify the exact back: `completely open bare back with only a single thin halter strap to a tie-neck knotted at the nape, NOT racerback NOT scoop-back NOT crossed straps`. |
| side-profile holding a bag/prop with the near arm | Arm bends backward / wrist twists unnaturally (broken elbow-to-hand line) to keep the prop in frame. | Pose the arm `hanging straight down and slightly FORWARD in front of the thigh, natural straight elbow, forearm and wrist in a natural anatomical line NOT twisted NOT bent backward`. Validated black cowl NYE v1 (2026-06-03). |

### Concrete before/after examples

**180° flip fail (pink café v3 slide_04 first attempt):**
- ❌ `body turned three-quarter away from camera, face looking back over right shoulder`
- ✅ `body turned three-quarter facing slightly right toward camera, head turned over right shoulder looking at camera`

**Hand-on-closure fail (pink café v3 slide_05 first attempt):**
- ❌ `right hand lightly touching ribbon tie at neckline`
- ✅ `right hand raised lightly to cheek with fingers near jawline`

---

## 9. Partial rerun procedure (replacing specific slides)

When only 1-2 slides fail review. The danger here is off-by-one indexing — overwriting good slides with replacements meant for bad ones.

**Procedure:**

1. **Identify exact problem slide numbers.** Read the source carousel folder's files in order. Confirm which slide index (00-05) has the issue. Write it down.
2. **Create `_<name>_fix.txt`** containing only the slides to regenerate. Comment which target slide index each replacement maps to.
3. **Run with separate name** to avoid clobbering the main folder:
   ```
   --name <carousel>_fix --flux-dev --kontext
   ```
4. **Map fix outputs to targets explicitly** before any copy. Write the mapping as a 2-column table:

   | Fix output | Target in main folder |
   |---|---|
   | `<carousel>_fix/slide_00_cand_0.png` | `<carousel>/slide_04_cand_0.png` |
   | `<carousel>_fix/slide_01_cand_0.png` | `<carousel>/slide_05_cand_0.png` |

5. **Copy with explicit destination filenames.** Never use wildcards or sequential names — type the destination filename in full.
6. **Read each destination file after copy** to verify the correct image landed. (Use the Read tool in Claude Code.)

**Common failure mode (this session):** writing "slide_03 replacement" and "slide_04 replacement" in the fix file when the actual problem slides were 04 and 05 → off-by-one. Always cross-reference the source folder, not the fix file's comments.

### Post-process-only rerun (skip FLUX, replay ReActor + hand + skin lock)

When the issue is in the post-process pipeline (skin lock burn, hand artefact, etc.) and the FLUX renders themselves are good, use `scripts/reprocess_carousel_post.py` instead of full Stage B. It reads the pre-faceswap base files from `_intermediate/` and re-applies Stages 3 → 3.5 → 3.6 in place.

```powershell
python scripts/reprocess_carousel_post.py `
  --carousel-dir output/YYYY-MM-DD/ananya/carousel_<name>/
# optional flags:
#   --face-ref <path>   (default: character/ananya/seeds_v2/face_ref_v2.png)
#   --cand N            (candidate index, default 0)
#   --seed-base N       (reroll hands without renaming files; default 0 = deterministic)
```

- ~5-8 min for 6 slides vs ~12-15 min for full Stage B.
- Hand-detail seed is deterministic (`sha256(base_filename) + seed_base mod 2^31-1`). Same carousel reprocessed twice produces identical hands. Bump `--seed-base 1` to reroll without renaming files.
- **In-place overwrite of slide_*.png** — if you need to keep the original, copy the carousel folder first (`cp -r <folder> <folder>_backup`).
- Validated 2026-05-25 on orange tank cafe v3 (skin lock feather patch reroll) and pink corset cafe v3 (smoke test, no regression on low ΔL).

---

## 10. Caption workflow

- **Location:** `character/ananya/captions/<carousel_name>.txt` — one file per carousel, same name stem as the anchor YAML and prompt file. **Also copy it as `caption.txt` into the carousel's output folder** (`output/YYYY-MM-DD/ananya/carousel_<name>/caption.txt`) so the post text ships next to the images.
- **Format:** opening hook line (lowercase, casual tone) → neighborhood location tag → `#AI` disclosure (mandatory) → max 5 hashtags.
- **Lowercase tone** throughout (per `feedback_instagram_captions.md`).
- **Neighborhood-level location tags**, not landmark, not city-level. (`Bandra` not `Mumbai`; `Lodhi Colony` not `Lodhi Garden`.)
- **Rotate hashtags** across posts — don't reuse the same 5 every time.
- **Write the caption AFTER carousel validation passes.** The hook should reflect what actually rendered, not what you hoped would render.

---

## 11. Review checklist — 7 goal criteria

Apply to every slide. Fail if any criterion fails on any slide (with documented exceptions).

| Criterion | Pass | Fail |
|---|---|---|
| Face vs face_ref_v2.png | Same facial structure, warm tone, dark eyes, full lips — recognizable as same person | Face shape changed, eye color shifted, wrong skin tone |
| Body M-size hourglass | Defined waist + natural curves, not slim editorial, not plus-size | Slim-editorial drift, plus-size rendering |
| Skin tone | Body skin matches face_ref cheek tone (ΔE < 5 after lock) | Visible mismatch between face and arms/legs/décolletage |
| Skin lock seam | Body silhouette blends smoothly into BG with no edge halo | Visible bright halo / chromatic ring along arm/shoulder/leg silhouette — check post-process log: if ΔL > 10 and burn visible, feather may have failed (verify `_MASK_FEATHER_SIGMA = 8.0` in `scripts/skin_color_match.py`); reprocess via `scripts/reprocess_carousel_post.py` |
| BG consistency | Same scene tokens as anchor (arch, column, foliage etc) across all slides | New scene invented on any slide |
| Accessories | Earrings/bracelet present on most slides; outfit core hold all | Outfit core changed (neckline, strap detail, color) |
| Hair | Long dark loose wavy on every slide | Hair length, color, or style changed |
| Pose variance | 6 distinct compositions (not 6 variants of same pose) | Two or more slides have visually-similar pose |
| Hand realism | All visible hands have 5 fingers, natural joint geometry, no fused fingers, no extra thumbs | Visible 6-finger, fused, or warped hand — Stage 3.5 hand detail occasionally misses on cup-grip / on-railing / fingers-spread poses. **Hands resting on a white/light skirt fuse every time** (low contrast → YOLO finds no bbox → Stage 3.5 never fires; `behind back` does NOT fix it). Fix by re-posing both hands above the bust on skin/hair/darker drape and regenerating the slide — post-process alone cannot repair a fused hand. See §8 row. Validated white off-shoulder lace corset v1 (2026-05-30). |
| Hand count | Exactly 2 hands visible per full-body slide (0-2 in closeups) | More than 2 hands — Kontext-hallucinated third arm/hand. Common triggers: raised arm + hand-at-hip + dupatta-drape combos; lean-against-object poses where prompt names two hand positions but Kontext adds a hair-touch (validated red chikankari lehenga v9d slide_03: 3 hands = hair-touch + chest + hip). Mitigation: write the prompt with explicit `BOTH arms hanging straight at sides, NO raised arms, NO hands at chest, NO third arm, only two hands visible total` |

### Documented exception

- Accessories (specifically bracelets, rings, footwear) may drop in/out across slides — Kontext limitation. Acceptable as long as the outfit core (neckline, fit, color, fabric) holds across all 6.

### Failure-resolution modes (when 1 slide fails)

When 1 slide fails review (hand deformity, pose duplicate, etc.) and other 5 pass, pick the cheapest fix:

1. **Drop the slide, ship 5-slide carousel.** Instagram carousels accept 2-10 slides. Often the simplest path — note dropped slide in the caption file header. Use when the dropped slide doesn't carry a critical pose (hook / closeup).
2. **Reroll only the post-process** (`scripts/reprocess_carousel_post.py --seed-base 1` → bumps deterministic hand seed). Use when failure is in Stage 3.5 hand detail or Stage 3.6 skin lock and the FLUX render itself is fine. ~5-8 min for all 6 slides (overwrites all but result for unaffected slides should be identical).
3. **Partial rerun with `_<name>_fix.txt`** (per §9 partial rerun procedure). Use when the FLUX render itself is bad (pose duplicate, BG drift). ~3 min per slide. Copy with explicit destination filename.

Validated 2026-05-27 on black puff crop haveli v1: slide_00 had hand deformity, user shipped 5 slides (option 1) — accepted as cheapest fix when the hook slide had a usable alternative in slide_04.

---

## 12. Worked examples

### Red bandeau choli + lehenga, indoor festive carved-door v9d (2026-05-29)

- **Anchor YAML:** `character/ananya/anchor_libraries/red_chikankari_lehenga_festive.yaml`
- **Prompt file:** `character/ananya/carousel_prompts/red_chikankari_lehenga_festive.txt`
- **Caption file:** `character/ananya/captions/red_chikankari_lehenga_festive.txt`
- **Output:** `output/2026-05-29/ananya/carousel_red_chikankari_lehenga_v9d/` (slide_03 replaced from `_v9d_fix/` per §9)

**Recipe:** Body LoRA 0.5, seed 334521876, face_ref_v2, `--flux-dev --kontext`, premium tier (deep cleavage + bare midriff), face+body skin lock auto-applied.

**Outfit:** plain solid red sleeveless deep bandeau-neckline choli with cropped shoulder straps and tiny scattered white mukaish pearl-dot speckles, plain solid red low-rise flared lehenga skirt, sheer plain solid red chiffon dupatta over left shoulder, large gold jhumka tassel earrings with pearl drops, gold watch + bangles on right wrist.

**BG:** indoor festive setting with dark carved teak wooden door panel + soft pink-cream wall + traditional genda-phool wall hanging on left (cream/pink/green/yellow fabric beads + bells).

**Neckline iteration (9 anchor rounds before landing):** the carousel exposed a hard FLUX tradeoff between cleavage depth and shape geometry. Square + modest cleavage works (v5). Deep + V-cut works (v6). Deep + true square does NOT — FLUX biases toward V/sweetheart at deep cuts because that's the training distribution. The landing recipe was a low underbust bandeau-strip framing: `narrow thin underbust bandeau strip wrapped horizontally low under the breasts with shoulder straps stitched on top, the band top edge cuts across at low bust apex level exposing both breasts mostly above the band showing extreme deep cleavage`. This produced deep cleavage with a horizontal-ish top edge.

**Fabric correction:** the early "chikankari embroidered" wording rendered as large white motifs scattered across red — wrong. The refs were plain solid red satin with tiny scattered white mukaish pearl-bead dot speckles only. The fix was an explicit list of NOT clauses: `NO chikankari NO large embroidered motifs NO trim NO gold edging NO visible buttons NO button strip NO contrasting hem band`. Pattern: when refs read as "plain solid" with subtle texture, enumerate the negatives — FLUX biases toward visible embroidery on red Indian wear.

**Body push:** balanced push from §2 fabric table (fitted choli + voluminous skirt mix), plus an isolated bust boost (`body is M-size everywhere EXCEPT bust which is two full cup sizes larger and heavier than the rest of the build`). Bust larger without rest of body drifting plus-size — works.

**Skin lock face-inclusion fix (script patched mid-run):** warm festive lighting was making the rendered face darker than face_ref_v2 baseline (via ReActor blend bleed-through). With face previously excluded from the LAB shift, the body got lifted +12-18 L to match face_ref while the face stayed at the warm-cast tone → visible face/body mismatch. Patched `scripts/skin_color_match.py` to apply the LAB shift to the FULL skin mask (face + body) so both unify to the fair face_ref tone. Cap raised back to `_MAX_L_SHIFT = 25.0` / `_MAX_AB_SHIFT = 12.0` since the inclusion absorbs the larger lift cleanly. Rule: warm-cast scenes (festive, golden hour, indoor incandescent) need face-inclusion lift, not face-exclusion preservation.

**Three-hand fail and partial rerun:** v9d slide_03 prompt was `leaning shoulder against door + right hand on hip + left arm against column`, but Kontext hallucinated a raised hair-touch arm (likely bleed from slide_02's hair-push prompt) → rendered 3 hands (hair-touch + chest grip + hip). Fixed via partial rerun (`_red_chikankari_lehenga_fix.txt`) with explicit `BOTH arms hanging straight at sides, NO raised arms, NO hands at chest, NO third arm, only two hands visible total` and copied to `slide_03_cand_0.png`. New §11 hand-count criterion captures this failure mode for future review.

**ComfyUI port fallback:** during this session ComfyUI Desktop fell back to port 8001 because port 8000 was held by a zombie process. Updated `scripts/comfyui_api.py:find_comfyui_port` candidate list to include 8001. Rule: if a script reports "ComfyUI not running" but a port-8000 listener exists, check process age + restart ComfyUI; the script now also tries 8001 as fallback.

**Final state:** all 7 review criteria pass + hand realism + hand count + skin lock seam pass. slide_02 has partial back-to-camera (Kontext interpreted "head over shoulder toward camera" as side-profile away) but accepted as artsy variation. Carousel approved for posting.

### Black strapless bandeau + leather pencil maxi, event launch backdrop v5 (2026-05-26)

- **Anchor YAML:** `character/ananya/anchor_libraries/black_strapless_leather_maxi_event.yaml`
- **Prompt file:** `character/ananya/carousel_prompts/black_strapless_leather_maxi_event.txt`
- **Caption file:** `character/ananya/captions/black_strapless_leather_maxi_event.txt`
- **Output:** `output/2026-05-26/ananya/carousel_black_leather_maxi_event_v5/`

**Recipe:** Body LoRA 0.5, seed 334521876, face_ref_v2, `--flux-dev --kontext`, premium tier (strapless cleavage), patched feather skin lock auto-applied.

**Outfit:** fitted black stretch jersey strapless bandeau tube top, high-waisted black faux leather pencil maxi skirt (no slit, ankle-grazing), burgundy oxblood leather pouch with gold knot clasp, gold watch + small gold hoop earrings, black ankle-strap stiletto heels.

**BG:** indoor event launch backdrop, terracotta-red wall + matching terracotta-red floor (continuous installation), large illuminated white neon-tube generic retail logo sign on wall (illegible script avoids real-brand trademark risk).

**Body push iteration (5 anchor rounds before landing M-size hourglass):**
- v1 — full orange-tank push (`size 12 curvy ... fuller heavier ... thick fuller thighs touching`) → plus-size. Leather is slimming so push over-corrected.
- v2 — backed off to slim-default baseline (`M-size hourglass ... soft natural curves`) → slim editorial.
- v3 — balanced middle (`M-size hourglass with hips noticeably wider than waist + fuller chest + NOT slim NOT plus-size NOT oversized`) → landed M-size hourglass. New §2 fabric-aware table captures this rule for next time.

**Floor cleanliness iteration (v3 → v4):** bare `polished concrete or marble floor` rendered scuffed and stained. Adding `clean spotless ... no stains no scuffs no dirt no marks` enforced a clean glossy surface. New §5 floor cleanliness rule.

**Wall-floor colour match (v4 → v5):** v4 had grey marble floor that broke the immersive red event installation. Adding `painted same terracotta-red as the wall behind so wall and floor blend into one continuous uniform red event installation surface` rendered a red floor matching the wall. New §5 event-installation rule.

**Skin lock burn check:** ΔL reached 13.7 on slide_04. **No burn visible on any slide** — feather patch (σ=8) validated in production. Closeups + full-body all clean.

**Trademark note:** prompt requested `illuminated white neon-tube generic retail logo sign`. FLUX rendered illegible script (`Juime`, `Awive`, etc.) across slides — exactly the desired safe behaviour, no real brand reproduced.

**Final state:** all 7 review criteria pass + skin lock seam criterion (new §11 row) passes. Carousel approved for posting.

### Orange tank + brown wrap mini skort café v3 (2026-05-25)

- **Anchor YAML:** `character/ananya/anchor_libraries/orange_tank_brown_skort_cafe.yaml`
- **Prompt file:** `character/ananya/carousel_prompts/orange_tank_brown_skort_cafe.txt`
- **Caption file:** `character/ananya/captions/orange_tank_brown_skort_cafe.txt`
- **Output:** `output/2026-05-25/ananya/carousel_orange_tank_skort_cafe_v3/`

**Recipe:** Body LoRA 0.5, seed 334521876, face_ref_v2, `--flux-dev --kontext`, premium tier (low scoop), skin lock (patched feather) auto-applied.

**Outfit:** fitted orange ribbed tank bodysuit, thin string spaghetti straps, deep scoop neckline, chocolate brown high-waisted A-line mini wrap skirt with single left-hip lace-up corset tie, white baseball cap, brown leather crossbody.

**Body push that finally landed M-size hourglass on FLUX:** `size 12 curvy body with full M-size hourglass, fuller heavier curvier build with soft visible belly midsection, wide hips noticeably wider than waist, thick fuller thighs touching, distinctly NOT slim NOT editorial NOT model-thin`. Earlier v1/v2 attempts using only `fuller curvier body with soft visible midsection` produced slim-editorial drift. FLUX bias is strong — push the negative comparison hard (`NOT slim NOT editorial NOT model-thin`) and add concrete size token (`size 12`).

**Skort vs shorts fix:** `cotton A-line mini wrap skirt with smooth solid front skirt panel falling unbroken to mid-thigh and NO shorts underneath, single decorative ribbon lace-up corset-tie detail only on left hip side seam`. Earlier `mini skort with side ribbon lace-up tie` rendered as cuffed shorts with lace-up on both sides. Naming `wrap skirt` + `no shorts underneath` + `single side seam` flipped it.

**Skin lock burn fix (skin_color_match.py patched mid-run):** Stage 3.6 was producing visible halo at body silhouette on full-body slides (00, 02, 04) when ΔL > 10. Root cause: hard bool mask + flat LAB shift = seam at mask edge. Patched: feather mask with Gaussian blur σ=8px, alpha-blend the LAB shift. Reprocessed 6 slides via `scripts/reprocess_carousel_post.py` (reads `_intermediate/*_base.png`, redoes ReActor + hand + patched skin lock, ~5-8 min, skips expensive FLUX). Closeups (01, 05) were unaffected before patch — burn only visible on large-area body shots.

**Final state:** all 7 review criteria pass. Carousel approved for posting.

<!-- Pink corset café v3 (2026-05-24) pruned 2026-05-29 per §13 ≤3-most-recent rule.
     Summary preserved in flux_dev_rulebook.md Validated Results (RULE 15). -->

---

## 13. Doc maintenance rule

This doc decays the moment it stops matching reality. To keep it current:

### When to update

- A new failure mode is found (forbidden pattern, accessory class, outfit type)
- A new pattern works (new pose vocabulary, new outfit-type rule)
- A new Kontext quirk is discovered
- A script flag, default, or workflow stage changes
- A successful carousel reveals something novel — add it to pose vocabulary or worked examples

### Who updates

Claude proposes the diff inline **during the same session** that produced the learning. User approves. Never let a learning sit only in chat — it disappears at compaction.

### Where to add

| Learning type | Target section |
|---|---|
| New forbidden pattern | §8 forbidden patterns table |
| New successful pose | §7 pose vocabulary |
| New outfit-type rule (lehenga, gown, jumpsuit etc) | §4 per-outfit-type table |
| New BG quirk (rooftop, indoor, beach etc) | §5 BG consistency |
| New accessory limitation | §6 accessories |
| New worked example | §12 (keep ≤3 most recent; older move to `flux_dev_rulebook.md` Validated Results) |

### Versioning

Update the `Last updated:` line at the top every edit. Format: `YYYY-MM-DD — one-line summary of change`.

### Conflict resolution

If this doc contradicts `flux_dev_rulebook.md` or `feedback_prompt_cookbook.md`, **this doc wins.** Update the others to match or remove the conflicting content. This is the canonical procedural reference.

### Pruning

If a rule hasn't fired or been referenced in 3+ months and the underlying tool/script has changed, mark it `[DEPRECATED YYYY-MM-DD]` rather than deleting. Future debugging may need the history.

---

## 14. Ultra-realism pass (`--ultra`)

Optional selective realism refiner that adds genuine skin/hair/fabric micro-texture to kill the plastic/waxy AI look. **Additive and default-off** — without `--ultra` the pipeline runs the exact current path (zero behaviour change).

### What it does

Inserts a **Stage 2.5** between FLUX-Kontext gen and ReActor:

```
FLUX-Kontext gen → [NEW: selective realism pass] → ReActor → hand detail → skin lock
```

- Workflow: `workflows/realism_selective.json` — `UltralyticsDetectorProvider (segm/yolov8n-seg.pt)` → `ImpactSimpleDetectorSEGS` (person SEGS) → `DetailerForEach` on an **SDXL realism checkpoint (RealVisXL_V4.0)** at denoise **0.38**, re-rendering ONLY the subject's skin/hair/fabric. **Background is left untouched** (no global refine).
- Runs **BEFORE ReActor** → the final face is always `face_ref_v2` (ReActor applies last), so **identity is guaranteed** regardless of the realism pass. The face SEGS sub-pass in the workflow is bypassed for this reason.
- Adds ~30-60s/slide on RTX 3050 6GB. Degrades gracefully: on any failure the slide ships the plain FLUX base.

### Why selective, NOT global

A whole-image tiled refine (tested + rejected) dirties the **entire** image — adds vintage grain/noise to the background, not just the subject. The laplacian-variance metric LIES here (rewards noise, not real detail) — **judge visually**. Person-SEGS detail keeps the BG clean while adding real texture to skin/hair/fabric only. (See memory `feedback_realism_selective_detail`.)

### How to use

Global flag (all slides):
```powershell
python scripts/faceswap_carousel.py --anchor-config <...>.yaml --prompts <...>.txt --name <...> --flux-dev --kontext --ultra
```

Per-slide token in the prompt file (overrides the global flag):
- `ultra=true` — on at default denoise 0.38
- `ultra=false` — off (e.g. bulk/filler slides to save time)
- `ultra=0.44` — on AND override denoise to 0.44 (more skin pore detail; use for **hero closeups** where fabric is minimal)
- `ultra=0` — off

### Denoise tuning (validated 2026-06-06)

| Denoise | Result |
|---|---|
| 0.38 (default) | Sweet spot — skin pore + fabric texture, natural, BG clean |
| ~0.44 | More skin micro-texture; good for face/skin-dominant closeups |
| 0.50 | Over-processes fabric (crunchy seams/weave), edges toward AI-sharpened; not worth it as default |

### When the uplift is worth it

- **Most visible in bright daytime deep-focus** scenes — flat even light exposes smooth plastic skin, so the texture gain reads strongly.
- **Moderate in dark/neon** scenes — colored low light hides skin detail, so the gain is subtle at feed size (pronounced at full zoom).
- **Ceiling caveat:** ReActor's pasted face is still the realism limit on the **face itself**; ultra improves body skin/hair/fabric and the swap-seam blend, not the intrinsic face resolution.

### Validated carousels

- `white_bodycon_club_neon` (night neon) — uplift real but moderate; BG stayed clean, identity held.
- `black_tube_beige_cargo_daycafe` (daytime deep-focus) — uplift clearly visible; non-ultra legs/midriff showed CGI sheen, ultra rendered matte natural skin.

---

## 15. Lens / DOF profile (`lens_profile:` anchor YAML key)

Optional anchor-YAML key that appends a camera/depth-of-field snippet to the anchor prompt **and every slide prompt** (so framing stays consistent). **Only applied when explicitly set** — omit it and nothing changes (existing anchors that hand-write `shot on Sony A7IV…` are unaffected).

```yaml
lens_profile: editorial   # or: selfie
```

| Profile | Look | Use for |
|---|---|---|
| `editorial` | Sony A7IV 50mm f1.8, shallow DOF, **creamy background bokeh** | Produced OOTD carousels — subject pops off a blurred BG |
| `selfie` | iPhone wide ~24mm, **deep focus, everything sharp, NO bokeh**, arm's-length candid | Matches real influencer phone selfies — daytime/outdoor candids |

**Why it matters:** the bokeh→deep-focus swap is itself a realism lever. DSLR bokeh reads "produced/AI"; phone deep-focus reads candid-real. The `--ultra` pass (§14) then sharpens the now-in-focus background cleanly. Pair `lens_profile: selfie` + `--ultra` for the most photo-real daytime look.

Do NOT also hand-write lens text in the prompt when using this key — pick one (the key OR inline), not both, to avoid contradictory camera tokens.
