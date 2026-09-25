# AI Influencer

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python)](https://python.org)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Local-grey?style=flat)](https://github.com/comfyanonymous/ComfyUI)
[![License](https://img.shields.io/badge/License-Proprietary-red?style=flat)](LICENSE)

Local pipeline for generating photorealistic, face-consistent images of virtual AI influencer personas. Uses cloud-trained Kohya Dreambooth LoRAs to lock face identity across scenes, outfits, and lighting conditions.

**Primary character:** Ananya — North Indian fashion and lifestyle creator (trigger: `AnanyaAI`)

---

## How it works

1. Train a Dreambooth LoRA on seed images (Kohya, cloud)
2. Load LoRA into ComfyUI running locally on RTX 3050 6GB
3. Run generation scripts — face identity locked via trigger word
4. Auto-caption outputs with Claude (multimodal) for social posts
5. Audit identity consistency with CLIP similarity scoring

## Generation tiers

| Tier | Pipeline | Time | Use case |
|------|----------|------|----------|
| SDXL + LoRA | ComfyUI workflow | 30–60s | Primary — face-locked |
| SDXL + IP-Adapter | ComfyUI workflow | 45–90s | Identity + style ref |
| FLUX.1-schnell | ComfyUI workflow | 5–15 min | Text-only, no LoRA |
| FLUX.1-dev | ComfyUI workflow | 20–40 min/slide (RTX 3050) | Premium quality, multi-anchor carousel |
| Bootstrap | Seed generation | Varies | Pre-LoRA training seeds |

> **Note:** Never load SDXL LoRAs into FLUX workflows — architecturally incompatible, crashes ComfyUI.

## Key scripts

| Script | Purpose |
|--------|---------|
| `generate.py` | Single image generation |
| `batch_generate.py` | Batch from prompt file |
| `bootstrap_seeds.py` | Seed generation for LoRA training |
| `prepare_training_data.py` | Training dataset prep + captioning |
| `auto_caption.py` | Claude multimodal caption generation |
| `generate_carousel.py` | Carousel/reel generation |
| `clip_similarity_audit.py` | Identity consistency checks |
| `comfyui_api.py` | Reusable ComfyUI REST client |
| `mcp_server.py` | FastMCP server integration |

## Requirements

- **GPU:** NVIDIA RTX 3050 6GB VRAM minimum (tested), VRAM rescue mode for tighter fits
- **ComfyUI:** Running locally at `127.0.0.1:8000`
- **Python:** 3.x
- **LoRA weights:** Not included — train via Kohya Dreambooth (`AnanyaAI_v1_Prod.safetensors`)

## Setup

```powershell
# Install dependencies
pip install -r requirements.txt

# Set environment variables (PowerShell)
$env:HF_TOKEN = "your_huggingface_token"
$env:ANTHROPIC_API_KEY = "your_anthropic_key"  # pragma: allowlist secret

# Download models
python setup/download_models.py

# Start ComfyUI (separate terminal)
python ComfyUI/main.py --port 8000

# Generate
python generate.py
```

## Characters

| Character | Trigger | Status |
|-----------|---------|--------|
| Ananya | `AnanyaAI` | Production |
| KaviB | `KaviB` | Regression testing |

## Workflow injection

Workflows use sentinel strings (`_claude_inject_*` in `_meta.title`) as injection points rather than node IDs — node IDs change on every ComfyUI export. This makes workflows stable across ComfyUI updates.

---

© 2026 Kavya Barnadhya Hazarika. All Rights Reserved.
This repository is proprietary — see [LICENSE](LICENSE) for details.
## Realism ablation: skin and face pipeline

Run after installing `requirements.txt`, starting local ComfyUI and installing the
nodes/models used by `t2i_sdxl_lora.json` and `faceswap_reactor.json`:

```powershell
python scripts/realism_ablation.py --face-ref character/ananya/seeds_v2/face_ref_v2.png --seed 334521876 --prompt "woman in a white linen dress, natural daylight, full-body fashion photograph"
```

The script uses the configured Juggernaut XL v9 checkpoint, 30 SDXL steps,
896x1152 output and Ananya SDXL LoRA. Set `--port 8002` if auto-discovery
chooses the wrong local ComfyUI instance. `--face-ref` is the one identity source
for all swapped cells, not a scene/init image. Keep the same prompt, seed and
face reference for each comparison. Use a reference you have permission to use.

It writes 10 full-resolution images, a `manifest.json` and a labeled
`contact_sheet.png` under `output/realism_ablation/`. The default sheet contains
full native-resolution frames, with no scaling. To compare the same exact face
region across cells, pass `--crop X Y WIDTH HEIGHT` in pixels; it only crops
sheet tiles and does not alter the saved images. Use coordinates inside every
output image. Run the grid again with a different seed to check the finding.

The baseline is swap on, CodeFormer visibility 1.0, LoRA .9 and CFG 8. Each
other cell changes **one** axis: swap off; CodeFormer .5/off; LoRA .65/.75/.85;
CFG 5/6/7. Face and hand detailing remain on, with fixed seed and the same CFG
as the primary sampler. The CodeFormer-off cell still runs inswapper and simply
sets restoration visibility to zero. ReActor's CodeFormer weight stays .75.

This isolates the SDXL/face-swap stage only. It intentionally does not run
UltraSharp, skin-color matching, FLUX carousel assembly or video; those are
separate pipeline stages and need separate before/after comparisons. This is
not a 40-way factorial sweep and does not automate aesthetic selection. Inspect
skin pores, face seams, fabric, hands and lighting at 100%; keep the best cells
for a small reel motion test. If a run fails, completed cells remain on disk;
remove the output folder before starting a fresh run to avoid mixing runs.
