# Realism Ablation — Results (PR #158)

## Ablation v1 — invalid, discarded

`output/realism_ablation/` (first run) used a full-body, backlit, window-silhouetted
prompt. Face filled too little of the frame to judge skin/restore quality, and the
raw (pre-swap) generation didn't visually match `face_ref_v2.png` closely enough to
serve as a clean identity baseline. Kept on disk for the record; **not** the basis
for the verdict below.

## Ablation v2 — fair rerun

- **Input:** `character/ananya/seeds_v2/face_ref_v2.png` (well-lit, front-facing, face fills frame)
- **Prompt:** medium-close portrait, facing camera directly, soft even daylight from the front, cream sweater, plain wall background
- **Canvas:** 768×1360 (reels aspect, per `config.yaml` `reels:` block)
- **Seed:** 334521876 (fixed across all 10 cells)
- Grid: `output/realism_ablation_v2/contact_sheet.png`, raw cells `00–09_*.png`, `manifest.json`

### Verdict: **CodeFormer OFF** (`face_restore_model="none"`)

| Cell | Result |
|---|---|
| `swap OFF` (01_raw.png) | Different face entirely — broader nose, different jaw/brow than face_ref. Confirms ReActor swap is **mandatory** for identity lock; raw SDXL+LoRA output alone drifts. |
| `baseline` CodeFormer 1.0 (00_swap.png) | Identity locks correctly, but skin is over-smoothed — pores and fine texture flattened, reads "beauty-filtered" / plastic under close inspection. |
| `CodeFormer .5` (02_swap.png) | Middle ground — still visibly softer than natural skin, less plastic than 1.0. |
| **`CodeFormer OFF`** (03_swap.png) | Identity still locks correctly (ReActor swap unaffected by restore model). Skin retains natural pore-level microtexture — closest to "Instagram-real," not AI-smooth. **Winner.** |
| LoRA .65/.75/.85, CFG 5/6/7 | No meaningful skin-texture or identity difference vs baseline at this scale; not the deciding axis. |

**Bug fixed along the way:** the original "CodeFormer OFF" cell set `face_restore_visibility=0.0`,
but ReActor's node floor is `0.1` — ComfyUI rejected it with a 400. Fixed by setting
`face_restore_model="none"` instead (a real bypass, confirmed via `/object_info/ReActorFaceSwap`
on the live node), rather than clamping visibility to the floor. See `scripts/realism_ablation.py`
commit `38b66b7`.

**Situational note:** CodeFormer is not universally bad — it exists to fix genuinely degraded
source faces. On this pipeline (clean SDXL+LoRA generation, real ReActor swap, good source
face_ref), it has nothing to repair and only costs texture. Revisit CodeFormer ON if a future
target image has actual face artefacts pre-swap (heavy blur, compression, occlusion).

---

## Step 2 — Final outputs

Generated with `scripts/generate.py` (SDXL + `AnanyaAI_v1_Prod.safetensors` LoRA, seed
`334521876`), then identity-locked with `scripts/apply_faceswap.py` (new script, this PR)
using the winning config (`face_restore_model="none"`), then skin-tone corrected with
`scripts/skin_color_match.py` against `face_ref_v2.png`.

**Finding along the way:** `generate.py`'s default SDXL+LoRA workflow (`t2i_sdxl_lora.json`)
has no ReActor node — it relies on the trained LoRA alone for identity, which drifts
under different lighting/hair-state prompts (confirmed: image 1's first draft had a
visibly different face/hair/skin-tone than `face_ref_v2.png`). Applying ReActor
faceswap as a second pass (same as the carousel pipeline does) fixed this. Standalone
`generate.py` output should not be treated as identity-locked on its own for anything
customer-facing — it needs the faceswap pass.

| Output | Scene | Notes |
|---|---|---|
| `final_01_balcony_emerald_skinfix.png` | Balcony, golden hour, emerald wrap top | Minor warm hair-highlight drift from backlight (documented pitfall); face/skin corrected and consistent. |
| `final_02_cafe_linen_skinfix.png` | Cafe window, oversized linen shirt | Cleanest identity match of the three; hair-color lock prompt held (no drift). |
| `final_03_rooftop_terracotta_skinfix.png` | Rooftop, sunset, terracotta wrap dress | Three-quarter angle, no hands in frame (sidesteps hand-defect risk), strongest editorial composition. |
| `reels/anchors/final_reel_rooftop_skinfix.png` | Rooftop reel anchor, 768×1360 | Same scene/seed family as image 3, regenerated at reels canvas. Neckline reads deeper than "medium" here — flagged, acceptable for one premium hero shot. |
| `reels/rooftop_terracotta_reel.mp4` | Ken Burns push, 5s, pure zoom (no parallax/sway) | Built from the anchor still via `reel_parallax.py`. Verified via extracted frames — clean push, no warping, identity holds across frames. |

### Consistency check

All three stills plus the reel anchor share: seed `334521876`, face swapped from the
same `face_ref_v2.png`, and skin tone corrected to the same target LAB
(`61.2, 14.0, 17.1`, face_ref's cheek tone). Face shape, nose, brow, and lip shape
read as the same person across all four images on direct visual comparison. Hair
color holds dark brown in 2 of 3 stills; image 1 has mild warm rim-light drift,
noted above, not re-rolled (lighting artifact, not a LoRA/seed problem).

### What produced the final images

SDXL (Juggernaut XL) + `AnanyaAI_v1_Prod` LoRA at strength 0.9, seed `334521876`, 30
steps, CFG 8 — generated via `generate.py`. Each raw output was then passed through
ReActor faceswap (`apply_faceswap.py`, `face_restore_model="none"`, the ablation's
winning config) against `face_ref_v2.png`, and finally through `skin_color_match.py`
to lock body/face skin tone to the reference's LAB cheek value. The reel anchor used
the same pipeline at the 768×1360 reels canvas (`--reel-anchor`), and the final .mp4
is a pure 6% Ken Burns zoom-in with no parallax, rendered by `reel_parallax.py`.
