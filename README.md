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
| FLUX.1-schnell | ComfyUI workflow | 5–15 min | Text-only, premium quality |
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

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN=your_huggingface_token
export ANTHROPIC_API_KEY=your_anthropic_key

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
