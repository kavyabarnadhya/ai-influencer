# Ananya Carousel Workflow — Canonical Reference

**Last updated: 2026-05-25 — skin lock Stage 3.6 now Gaussian-feathers body skin mask (σ=8px) before LAB shift, eliminates seam halo at body silhouette when ΔL > 10. Added `scripts/reprocess_carousel_post.py` for re-running ReActor + hand detail + skin lock from `_intermediate/*_base.png` without re-rolling FLUX. Added orange tank cafe v3 worked example.**

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
| Skin tone post-process | Auto-applied | `scripts/skin_color_match.py` | Runs after hand detail. Locks body skin to face_ref cheek LAB. Mask Gaussian-feathered σ=8px before LAB shift — required to avoid seam halo at body silhouette when ΔL > 10 (validated on orange tank cafe v3). No prompt action needed. |

### Hard NEVER rules

- **NEVER add explicit skin tone tokens** (`warm medium-brown`, `tan complexion`, `olive`, `medium South Asian skin`). The seed + `South Asian woman` derives the correct tone. Adding tokens overrides the seed → wrong/darker tone.
- **NEVER re-prompt anatomy** (face shape, eye color, hair color, ethnicity, age beyond `23-year-old`). The seed + ReActor handle these. Re-prompting causes "double-baking" → distorted face.
- **NEVER swap the body seed** mid-project. `334521876` is the validated reference. Other seeds (`837492016`, `112847593`) produced wrong body types and are blocklisted.

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

### Per-outfit-type rules

| Outfit type | Key prompt rules |
|---|---|
| Ethnic (kurta, saree, lehenga) | `fitted/tailored/cinched` — NEVER `flowing/loose`. Dupatta needs contrasting color + named shoulder. Pallu pinned at left shoulder + falling behind. |
| Form-fitting western (slip, bodycon, mini frock) | Add `fabric shows soft midsection` to avoid slim-editorial default. |
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
- **All toward-camera** body angles (full or three-quarter). Body turned away (180° flip) breaks BG.

---

## 8. Forbidden Kontext slide prompt patterns

| Pattern | Why it breaks | Replacement |
|---|---|---|
| `body turned away from camera` / `back to camera` | Full 180° flip → Kontext repaints scene geometry → BG collapses, invents new scene | `three-quarter angle facing slightly left/right toward camera, head turned to camera` |
| `hand touching ribbon tie` / `fingers on lace` / `hand on button/zipper/strings` | Kontext reads as untying/opening the garment | `hand to cheek with fingers near jawline`, `fingertips lightly at collarbone`, `hand raised near shoulder` |
| `sitting` in a standing carousel | Structural pose change → BG + outfit drift; Kontext fights its own preservation logic | Separate carousel entirely; sitting = standalone post |
| `change to waist-up portrait framing` | Kontext ignores waist-up cue, stays full-body | `change to chest-up portrait framing showing face neck shoulders and neckline only` |
| `mid-stride walking toward camera` / any motion verb from a static anchor | Kontext preserves anchor composition — motion instruction silently ignored → slide becomes near-duplicate of anchor pose | Use static-friendly action: `stepping forward with one foot ahead`, `cup raised to lips taking a sip`, `head tilted to one side looking off-camera` |
| `hand raised through/pushing hair` combined with `holding object in other hand` | Three-hand artefact — Kontext renders the raised arm but also keeps an "expected" hand at the side, producing extra limb | Pick one: either hair-push (both hands free from objects) OR holding object (no hair-push). If both poses needed, split across two slides. |

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

---

## 10. Caption workflow

- **Location:** `character/ananya/captions/<carousel_name>.txt` — one file per carousel, same name stem as the anchor YAML and prompt file.
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
| BG consistency | Same scene tokens as anchor (arch, column, foliage etc) across all slides | New scene invented on any slide |
| Accessories | Earrings/bracelet present on most slides; outfit core hold all | Outfit core changed (neckline, strap detail, color) |
| Hair | Long dark loose wavy on every slide | Hair length, color, or style changed |
| Pose variance | 6 distinct compositions (not 6 variants of same pose) | Two or more slides have visually-similar pose |

### Documented exception

- Accessories (specifically bracelets, rings, footwear) may drop in/out across slides — Kontext limitation. Acceptable as long as the outfit core (neckline, fit, color, fabric) holds across all 6.

---

## 12. Worked examples

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

### Pink corset café v3 (2026-05-24)

- **Anchor YAML:** `character/ananya/anchor_libraries/pink_corset_mini_cafe.yaml`
- **Prompt file:** `character/ananya/carousel_prompts/pink_corset_mini_cafe.txt`
- **Caption file:** `character/ananya/captions/pink_corset_mini_cafe.txt`
- **Output:** `output/2026-05-24/ananya/carousel_pink_corset_cafe_v3/`

**Recipe:** Body LoRA 0.5, seed 334521876, face_ref_v2, `--flux-dev --kontext`, skin lock auto-applied.

**What worked:** Hand-on-hip slide_00 hook, 2 chest-up closeups (slide_01 direct gaze + slide_05 hand-to-chin), walking slide_02, three-quarter hair-push slide_03, off-camera relaxed slide_04.

**What was fixed mid-run:**
- v1 attempt: ran without `--kontext` → all 6 slides same neutral stand. Fixed by adding `--flux-dev --kontext` and rewriting prompts with explicit pose variety.
- v3 slide_04 first attempt: `body turned away from camera` → invented dark gate + hedge wall BG. Fixed by changing to "three-quarter facing toward camera".
- v3 slide_05 first attempt: `hand touching ribbon tie` → looked like she was opening the dress. Fixed by changing to `hand to cheek with fingers near jawline`.
- Slide_03 / slide_04 ended up identical after a partial-rerun off-by-one error. Fixed by generating a third distinct slide (full body off-camera candid) and copying to slide_04 with explicit filename mapping.

**Final state:** all 7 review criteria pass. Carousel approved for posting.

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
