# Reel deliverable — status: blocked, no local img2vid available

## Verdict

**No local image-to-video model is available on this machine.** Checked and confirmed
2026-09-25/26:

- `GET /object_info` on the running ComfyUI shows node *classes* for AnimateDiff,
  SVD, LTX, Hunyuan, Wan (`WanImageToVideo`, `SVD_img2vid_Conditioning`,
  `LTXVImgToVideo`, etc.) — these come from installed custom node packs, but a node
  class existing does not mean a model checkpoint exists.
- `models/unet/` and `models/checkpoints/` contain **zero** video model weights —
  only the four FLUX checkpoints (`flux1-dev-Q4_K_S.gguf`,
  `flux1-kontext-dev-Q4_K_S.gguf`, `flux1-schnell-Q3_K_S.gguf`,
  `flux1-schnell-Q4_K_S.gguf`). Nothing to actually run img2vid with.
- Most of the "video" nodes present are paid cloud-API wrappers anyway
  (`KlingImageToVideoWithAudio`, `RunwayImageToVideoNodeGen3a`,
  `MinimaxImageToVideoNode`, `Wan2ImageToVideoApi`, etc.) — not local inference.
- Hardware: **RTX 3050, 6GB VRAM, ~15.7GB system RAM** (confirmed via `nvidia-smi`
  and `Get-CimInstance Win32_ComputerSystem` this session) — unchanged from the
  constraint documented in `character/ananya/reels_deferred.md` (2026-05-27):
  6GB VRAM is below the practical floor for every local I2V path (Wan 2.1/2.2, SVD,
  LTX, Hunyuan, AnimateDiff-SDXL), and 16GB RAM is below the ~24GB de-facto minimum
  for CPU-offloaded T5-XXL (Wan).

This matches `reels_deferred.md`'s prior conclusion exactly — nothing has changed
since May that would unlock a local path.

## What's needed to unlock real-motion reels

Pick **one**:

1. **Hardware upgrade** — 12GB+ VRAM (RTX 4060 Ti 16GB or better, or a used 3090
   24GB) **and** 32GB+ system RAM. At that tier, local Wan 2.2 I2V becomes reliable
   and the existing identity-lock stack (per-frame ReActor + RIFE interpolation,
   already designed in `reels_deferred.md`) can run.
2. **Paid cloud budget** — as little as $5-15/month on fal.ai (Wan 2.2 I2V,
   ~$0.20-0.40/clip) covers a weekly cadence without local hardware changes.
3. **A plan upgrade that includes video** — ChatGPT Plus (Sora I2V, ~$20/mo, if
   available in-region) or Google AI Pro/Ultra (Veo 3 via Gemini).
4. **A genuinely new low-VRAM model** — nothing found as of this session; worth a
   recheck of kijai's GitHub / r/StableDiffusion / ComfyUI release notes next
   quarter for a distilled 6GB-viable I2V model.

Per the CLAUDE.md rule ("pan/zoom over a still is NOT acceptable" for this
deliverable), I have **not** substituted `reel_parallax.py`'s Ken Burns push here —
that tool remains valid for IG Stories only (per its own documented scope), not for
a "real motion" grid Reel. Shipping it here would misrepresent it as something it
isn't.

## What I did ship instead

The 5-image carousel (this same batch, `output/2026-09-26/ananya/restart/`) —
achievable with the existing FLUX + Kontext + ReActor pipeline, no video model
required. See the batch's own README/caption for details.

**Recommendation:** if reel cadence becomes a growth blocker, revisit trigger #6 in
`reels_deferred.md` applies — the $5-15/mo fal.ai path is the lowest-friction unlock
and doesn't require new hardware.
