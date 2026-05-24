# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A fully local, CLI-driven image generation pipeline for creating photorealistic, face-consistent images of virtual AI influencer personas using ComfyUI. Primary character: **Ananya** (North Indian fashion/lifestyle creator, trigger word `AnanyaAI`). Secondary/regression character: **KaviB** (trigger word `KaviB`). Face identity is locked via cloud-trained Kohya Dreambooth LoRAs.

**Platform:** Windows 11, RTX 3050 6GB VRAM, ComfyUI Desktop at `C:\Users\barna\Documents\ComfyUI` (port 8000)

## Commands

All Python scripts require the venv activated first:

```powershell
# Activate venv
.venv\Scripts\activate

# Verify setup (models + workflow sentinels + ComfyUI API)
python setup\verify_setup.py

# Generate a single image (default character: ananya)
python scripts\generate.py --prompt "rooftop portrait, golden hour, silk saree"
python scripts\generate.py --prompt "..." --character kavib

# Generate with low-VRAM rescue mode (768x1152, 24 steps, no FaceDetailer)
python scripts\generate.py --prompt "..." --rescue

# Batch generate from prompt file
python scripts\batch_generate.py --prompts character\ananya\lifestyle_prompts.txt --count-per-prompt 3
python scripts\batch_generate.py --prompts character\ananya\premium_prompts.txt --category premium

# Generate seed images for LoRA training (run once per mode)
python scripts\bootstrap_seeds.py --character ananya --mode closeup --count 16
python scripts\bootstrap_seeds.py --character ananya --mode medium --count 16
python scripts\bootstrap_seeds.py --character ananya --mode fullbody --count 16

# Prepare training dataset
python scripts\prepare_training_data.py --character ananya --validate
python scripts\prepare_training_data.py --character ananya --caption-style sdxl
python scripts\prepare_training_data.py --character ananya --zip-only

# First-time setup (Windows PowerShell)
powershell -ExecutionPolicy Bypass -File setup\install_windows.ps1
python setup\download_models.py --hf-token YOUR_HF_TOKEN
```

There are no automated test or lint commands — `setup/verify_setup.py` is the smoke test.

## Architecture

### Characters

| Character | Trigger Word | LoRA | Status |
|-----------|-------------|------|--------|
| Ananya | `AnanyaAI` | `AnanyaAI_v1_Prod.safetensors` | Primary (production) |
| KaviB | `KaviB` | `KaviB_v1_Prod.safetensors` | Secondary (regression test) |

Character configs live in `config.yaml` under `characters:`. Each character has: `trigger_word`, `lora`, `lora_strength`, `base_prompt_file`, `output_subdir`, `seeds_dir`. All scripts accept `--character [ananya|kavib]` (default: `ananya`).

Character files:
- `character/ananya/` — base_prompt.txt, lifestyle_prompts.txt, premium_prompts.txt, character_bible.md, seeds/
- `character/kavib/` — base_prompt.txt, lifestyle_prompts.txt, intimate_prompts.txt, character_bible.md, seeds/

### Generation Tiers

| Tier | Workflow | Speed | Identity-Locked |
|------|----------|-------|----------------|
| SDXL + LoRA | `t2i_sdxl_lora.json` | 30–60s | Yes (primary) |
| SDXL + IP-Adapter | `t2i_ipadapter.json` | 45–90s | Yes + style ref |
| FLUX.1-schnell | `flux_schnell.json` | 5–15 min | No (text-only) |
| Bootstrap (pre-LoRA) | `bootstrap_seeds.json` | 30–60s | No |

**SDXL LoRA tensors are incompatible with FLUX UNet** — never load a character LoRA into a FLUX workflow; it will crash ComfyUI.

### Workflow Injection System

Scripts inject values into ComfyUI workflow JSONs by matching `_meta.title` sentinel strings rather than node IDs (which change on each export). The sentinels are:

- `_claude_inject_prompt` — positive text prompt
- `_claude_inject_negative` — negative prompt
- `_claude_inject_seed` — seed, steps, CFG
- `_claude_inject_checkpoint` — model selection
- `_claude_inject_lora` — LoRA path + strength
- `_claude_inject_latent` — width × height

`verify_setup.py` fails fast if any sentinel is absent from a workflow file. When editing workflow JSONs, preserve these `_meta.title` values.

### ComfyUI API Client (`scripts/comfyui_api.py`)

REST polling over HTTP (no WebSocket). Submits workflow to `/prompt`, polls `/history/{prompt_id}` with exponential backoff, and downloads images from `/view`. ComfyUI must be running on `127.0.0.1:8000` before any script is called.

### Central Config (`config.yaml`)

All paths, model filenames, GPU flags, and generation defaults live here. Scripts read this file at startup — do not hardcode paths or filenames in scripts. Character-specific settings (lora, trigger word, prompts path, output subdir) live under `characters:`.

## Identity & Prompt Rules

**Critical — violating these causes distorted faces:**

1. The character trigger word must always be the **first token** in every post-LoRA prompt (`keep_tokens=1` in Kohya config).
2. **Never re-prompt anatomy** (face shape, eye color, hair, ethnicity, skin tone). The LoRA encodes these permanently.
3. Only prompt: clothing, setting, pose, lighting, mood, action.

Violations cause "double-baking" — the LoRA tries to overlay identity descriptions onto an already-encoded face, producing plastic or distorted results. The base_prompt.txt for each character contains the full identity description used only during pre-LoRA bootstrap.

## Carousel Pre-Flight (MANDATORY)

**Before running ANY Ananya carousel, complete this checklist. Skipping any step has produced documented failures.**

### 5-step pre-flight ritual

1. **Read `character/ananya/carousel_workflow.md` in full.** Not skim — read. The canonical procedural reference. CLAUDE.md is the gate; workflow doc is the full reference with worked examples.
2. **Confirm anchor YAML has identity locks:**
   - `face_ref: character/ananya/seeds_v2/face_ref_v2.png`
   - `anchor_seed: 334521876`
   - `anchor_body_lora_strength: 0.5` (or `0.0` for headshot / flowy linen)
3. **Confirm command includes `--flux-dev --kontext`** flags. Without these, FLUX img2img locks anchor composition → all 6 slides become the same pose.
4. **Always run `--anchor-only` first.** Wait for user approval. Then run full carousel. Never skip the gate.
5. **Read slide prompt file end-to-end** and verify no forbidden patterns (table below) appear.

### Mandatory run command templates

**Stage A — anchor only (always first):**
```powershell
python scripts/faceswap_carousel.py `
  --anchor-config character/ananya/anchor_libraries/<name>.yaml `
  --name <name>_v1 --flux-dev --kontext --anchor-only
```

**Stage B — full carousel (only after user approves anchor):**
```powershell
python scripts/faceswap_carousel.py `
  --anchor-config character/ananya/anchor_libraries/<name>.yaml `
  --prompts character/ananya/carousel_prompts/<name>.txt `
  --name <name>_v1 --flux-dev --kontext
```

### Forbidden Kontext slide prompt patterns

| Pattern | Replacement |
|---|---|
| `body turned away from camera` / `back to camera` | `three-quarter angle facing slightly left/right toward camera, head turned over shoulder` |
| `hand touching ribbon tie / lace / button / zipper / strings` | `hand to cheek`, `fingertips at collarbone`, `hand raised near shoulder` |
| `sitting` in a standing carousel | Separate carousel entirely |
| `change to waist-up portrait framing` | `change to chest-up portrait framing showing face neck shoulders and neckline only` |

### Partial rerun rule

Before copying any fix output into the source folder: **re-read the source carousel folder** to confirm exact slide indices that need replacement. Write the mapping as a 2-column table (fix output → target filename) BEFORE running copy. Verify each destination by reading the file after copy.

### Identity & post-process locks (auto-applied — do not override)

- **Hand realism**: `workflows/flux_hand_detail.json` runs as Stage 3.5 after ReActor — SDXL inpaint on YOLO-detected hand bboxes (`hand_yolov8s.pt`). Fixes FLUX 6-finger / deformed-hand artefacts. Adds ~10-15s/slide. Degrades gracefully on failure (ships original hands).
- **Skin tone**: `scripts/skin_color_match.py` runs after hand detail. Locks body skin to face_ref cheek LAB tone. NEVER add skin tone tokens to prompts.
- **NEVER re-prompt anatomy**: face shape, eye color, hair color, ethnicity, skin tone. Seed + ReActor handle these.

Full procedural reference + worked examples + per-outfit-type rules + caption workflow + maintenance rules: `character/ananya/carousel_workflow.md`.

## Content Compliance

- All outputs → `output/YYYY-MM-DD/{character}/` (character-specific subfolder)
- Ananya content: Instagram-first. Free tier: fully clothed. Premium/subscription tier: cleavage-visible editorial fashion (intentional — training data includes these images)
- Every post must carry `#AI` disclosure
- Age representation: **always 23** in all captions, bios, and prompts — never change this
- Do not use real-person likenesses in training data without explicit written consent
- Verify Juggernaut XL license terms before any commercial monetization

## LoRA Training Workflow (per character)

### v1 (complete — Kohya Dreambooth SDXL)
1. Run `bootstrap_seeds.py --character ananya` (3 modes × 16 images = 48 candidates)
2. Manually curate best 8 per mode → `character/ananya/seeds/{closeup,medium,fullbody}/`
3. Run `prepare_training_data.py --character ananya --validate && --caption-style sdxl`
4. Edit all `*.txt` captions — apply Isolation Rule (describe setting/pose/lighting only — never physical features)
5. Run `--zip-only` to produce `training_data_ananya.zip`
6. Upload zip + `setup/kohya_config.toml` to RunPod (~$5, ~45 min on RTX 3090)
7. Download trained `AnanyaAI_v1_Prod.safetensors` → `C:\Users\barna\Documents\ComfyUI\models\loras\`

### v2 (in progress — FLUX Dev LoRA via ai-toolkit)
Dataset: **33 curated images** in `character/ananya/seeds_v2/training_canonical/` — ready for captioning.
Trigger word for v2: `AnyV2X9` (different from v1 `AnanyaAI`)

Next steps:
1. `python scripts/auto_caption.py --input-dir character/ananya/seeds_v2/training_canonical --mode florence2`
2. Manually edit all 33 `.txt` files per `character/ananya/v2_scene_anchor_vocab.md`
3. `python scripts/clip_similarity_audit.py --input-dir character/ananya/seeds_v2/training_canonical`
4. `python scripts/prepare_training_data.py --character ananya --zip-only`
5. RunPod RTX A6000 + `ostris/ai-toolkit` → `AnanyaAI_v2_Prod.safetensors`

**Bootstrap images rejected:** `seeds_v2/experimental/bootstrap_2026-05-09_*/` — skin tone drifts lighter due to IPAdapter. Do not add to training set.

### faceswap_stock.py — selective runs
`--files` flag added 2026-05-13: pass comma-separated filenames to process a subset of `--input-dir`.
```powershell
python scripts\faceswap_stock.py --face-ref "..." --files "img1.jpg,img2.jpg"
```

See `setup/train_lora_guide.md` for full training details.

## VRAM Management

Target hardware: RTX 3050 6GB VRAM + 16GB system RAM.

- `config.yaml` flags: `use_lowvram: true`, `tile_vae: true`
- `--rescue` flag on `generate.py`: drops to 768×1152, 24 steps, disables FaceDetailer
- FLUX models available: `flux1-schnell-Q4_K_S.gguf` (6.3GB, preferred) and `flux1-schnell-Q3_K_S.gguf` (4.85GB fallback if OOM) — both in `Documents\ComfyUI\models\unet\`
