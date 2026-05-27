# Ananya Reels — Deferred (research + decision tree + revisit triggers)

**Status as of 2026-05-27:** real AI motion reels are deferred. CV-only Ken Burns push
shipped for IG Stories. Grid Reels parked until budget or hardware unlock.

This doc captures the research so future-you doesn't re-derive the same dead ends.

---

## What we tried this session

1. **CV motion (opencv + MiDaS depth + 2.5D parallax)** — built `scripts/reel_parallax.py`.
   - With depth-weighted parallax (`--depth-scale 0.6 --sway-px 18`): **face/hair melted at
     peak sway** due to UV displacement at depth boundaries. Fixed by clamping depth_scale to
     0..1 and gaussian-smoothing the depth map (σ=12).
   - With safe parallax (`--depth-scale 0.6 --sway-px 12 --dolly-px 6`): **silhouette wobble
     / cardboard-cutout effect** — subject slides as a rigid card inside a hole, uncanny.
   - With pure Ken Burns push (zoom-only, no parallax): **reads as PPT animation**. No
     internal subject motion. Audience pegs it as fake on grid Reels.
   - **Conclusion:** CV motion on portrait stills cannot fake internal motion (hair sway,
     breath, fabric flow). Ceiling reached.

---

## Hardware constraint (the binding fact)

- **Windows 11, RTX 3050 6GB VRAM (Ampere GA107), 16GB system RAM**
- **6GB VRAM is below the practical floor** for current image-to-video diffusion
- **16GB system RAM is below the 24GB de-facto minimum** for T5-XXL CPU offload (Wan)
- Both must clear before local AI video is reliable

---

## Why each local AI video path fails on this hardware

### Wan 2.1 / 2.2 (Alibaba) — researched, rejected

- **No official Wan 2.1 1.3B I2V** exists — 1.3B is text-to-video only
- Wan 2.1 14B I2V Q4 GGUF + BlockSwap fits ~5.5GB but **mid-render VRAM spike to ~30GB peak**
  on RAM-offloaded layers → 16GB RAM pagefile thrash → 25-40 min/clip
- Wan 2.2 5B I2V needs more VRAM than RTX 3050 has
- VACE 1.3B reference-image hack works but **motion fidelity is degraded** vs true I2V
- Cherry-pick rate **3-6 clips per keeper** for portrait subjects → ~2-4 hours wall-clock per
  usable 5s clip
- WanVideoWrapper 1.3.9+ broke many 6GB workflows (kijai issue #1644)
- Wan 2.1 is superseded by 2.2; 1.3B variant gets no updates
- Sources: kijai/ComfyUI-WanVideoWrapper issues #805, #1267, #1272, #1644, #1722, #1769, #1805

### SVD 1.1 (Stability) — researched, rejected

- Equally marginal on 6GB — same OOM tier as Wan, slower than expected
- **Stability removed SVD from API August 2025** → community moved on
- Face flicker, camera-biased motion (NOT hair sway), generic output, no prompt steering
- No published "verified working 6GB" workflow JSON in 2026
- Per-frame ReActor adds "swap shimmer" temporal artefact
- Palindrome reveals reverse tell on directional motion
- Sources: HF SVD discussions, kijai/ComfyUI-SVD repo

### LTX-Video / Hunyuan / CogVideoX / Mochi / AnimateDiff SDXL / Allegro

- All require ≥8GB VRAM (most 12GB+) — out of scope for RTX 3050 6GB
- LTX 2.x specifically reported needing 44GB **system RAM** at 6GB VRAM tier (we have 16GB)

---

## Why cloud was rejected this session

User stated: **no paid options.** Cloud paths exist but all gated:

| Service | Status |
|---|---|
| fal.ai Wan 2.2 I2V | Paid — ~$0.20-0.40/clip |
| Runway Gen-3 / Gen-4 | Paid — credits required |
| Sora (via ChatGPT Plus/Pro) | Paid — Plus ~$20/mo. **NOT in ChatGPT Go plan** (user verified) |
| Veo 3 (via Gemini) | Paid — Google AI Pro/Ultra ~$20/mo. User does NOT have Ultra plan |
| Kling AI | Free tier exists but extremely limited (user judged as effectively paid) |
| HailuoAI / Minimax | Has free credits but uncertain reliability; not verified by user this session |
| Pika Labs | Free credits very limited |
| Vidu | Free tier with watermarks + caps |

Free-tier paths universally have: watermarks, daily caps, queue waits, lower resolution,
ToS restrictions on commercial / AI-influencer content. None viable for sustained cadence
without paid uplift.

---

## What we did ship this session

- `scripts/reel_parallax.py` — Ken Burns push CV reel generator
  - Default: pure zoom-in (`--zoom 0.06`, no sway, no parallax)
  - Scope: **IG Stories only, NOT grid Reels** (PPT-tier on Reels)
  - Identity-safe: every pixel sampled from source PNG, no diffusion
  - Palindrome loop default for seamless cycling
  - ~5-10s render per clip
- `scripts/_extract_frames.py` — debug helper for visual review of mp4 output

---

## Revisit triggers (when to pick this back up)

Reopen reels when ANY of these change:

1. **Hardware upgrade** to 12GB+ VRAM (RTX 4060 Ti 16GB, used 3090 24GB, etc.) AND 32GB+
   system RAM. At that tier, local Wan 2.2 I2V becomes reliable.
2. **Budget unlock** for cloud — even $5-15/month on fal.ai covers a weekly cadence.
3. **ChatGPT Go upgrades** to include Sora image-to-video (watch OpenAI release notes for
   India market).
4. **Free tier opens up** — Kling/HailuoAI/Pika sometimes run promotional periods with
   higher free quotas. Recheck quarterly.
5. **New low-VRAM model** releases (something like a Wan 1B distilled I2V or a
   FramePack-style autoregressive that genuinely fits 6GB). Monitor: kijai's GitHub,
   r/StableDiffusion, ComfyUI release notes.
6. **Reels become carousel-growth blocker** — if analytics show carousel cadence is capped
   and Reels are the missing growth lever, the math on paid cloud changes.

---

## Decision tree for the next attempt

```
START
  │
  ├── Has budget for paid cloud? ($5-15/mo)
  │     YES → fal.ai Wan 2.2 I2V + per-frame ReActor post-process
  │     NO  → ↓
  │
  ├── Hardware: 12GB+ VRAM AND 32GB+ RAM?
  │     YES → local Wan 2.2 I2V + per-frame ReActor
  │     NO  → ↓
  │
  ├── Sora available in current ChatGPT plan? (verify monthly)
  │     YES → manual Sora workflow + post-process script
  │     NO  → ↓
  │
  ├── Free tier reliable on Kling/HailuoAI? (verify quarterly)
  │     YES → manual free-tier workflow + watermark crop + post-process
  │     NO  → ↓
  │
  └── Stay parked. Ship Ken Burns CV for Stories only.
```

---

## Identity preservation strategy (locked, applies to all future paths)

When local or cloud video is unlocked, the identity stack stays the same as the carousel
pipeline:

1. Generate video from carousel slide (Wan / Sora / Veo / Kling / cloud)
2. Per-frame ReActor swap using `character/ananya/seeds_v2/face_ref_v2.png`
3. RIFE 2× interpolation for smoothness
4. Optional: per-frame skin lock via `scripts/skin_color_match.py` (patched feather,
   see `carousel_workflow.md` §2)
5. Encode to 1080×1920 H264 mp4 for IG

Per-frame ReActor on 75 frames adds ~3-8s/frame on CPU (Stage 3.5 timings extrapolated)
→ ~5-10 min post-process per clip. Acceptable overhead.

---

## Cross-references

- `carousel_workflow.md` — canonical doc for stills pipeline (skin lock, hand realism,
  identity locks)
- `setup/ananya_flux_lora_reels.md` — original reels strategy doc (pre-dates this research,
  already says "outsource video to cloud" — consistent with this conclusion)
- `scripts/reel_parallax.py` — CV-only Ken Burns push generator (Stories scope only)
