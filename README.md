# AI Influencer Pipeline

A fully local, CLI-driven pipeline to generate photorealistic, consistent images of
an AI virtual female persona (KaviB) for Instagram, Telegram, and Fanvue.

**Hardware target:** RTX 3050 6GB VRAM · 16GB RAM · Windows 11

---

> **⚠ Commercial & Compliance Notice**
>
> Before monetizing any output from this pipeline:
> - **Model licenses:** Juggernaut XL's model card states it is *"not permitted behind API services."*
>   Contact RunDiffusion for a commercial license before monetizing at scale.
>   Verify the license of every model you use (FLUX, IP-Adapter, etc.).
> - **AI disclosure:** All public posts must carry an AI-generated content label
>   (e.g. `#AI` on Instagram, AI creator label on Fanvue).
> - **Age representation:** The KaviB persona is always stated as **23 years old**
>   in all captions, bios, and profiles. Never depict or imply any character is under 18.
> - **Adult content:** Restricted to platforms with explicit AI creator policies
>   (Fanvue, Telegram). **Never post adult content to Instagram.**

---

## How It Works

Face consistency is achieved via a **cloud-trained Kohya LoRA** (~$2–8, one-time):
1. Generate 24 seed images with Juggernaut XL (no LoRA yet)
2. Train a Dreambooth LoRA on RunPod or Civitai using trigger word `KaviB`
3. Download `KaviB_v1_Prod.safetensors` and drop it into ComfyUI
4. Every subsequent generation auto-prepends `KaviB` and loads the LoRA

No InsightFace or InstantID dependency. Face detail is enhanced by FaceDetailer
(YOLOv8 face detector via ComfyUI-Impact-Pack + Impact-Subpack).

---

## Generation Tiers

| Tier | Workflow | Speed | Face lock? |
|------|----------|-------|-----------|
| 1 — SDXL + LoRA | `t2i_sdxl_lora.json` | 30–60s | ✅ Yes (KaviB LoRA) |
| 1b — SDXL + IP-Adapter | `t2i_ipadapter.json` | 45–90s | ✅ Yes + style ref |
| 2 — FLUX Schnell | `flux_schnell.json` | 5–15 min | ❌ Text-only |

> **FLUX note:** SDXL LoRA tensors are incompatible with FLUX UNet architecture.
> Loading the LoRA into a FLUX workflow crashes ComfyUI. FLUX is for art/background
> shots only — use SDXL for all KaviB identity-locked images.

---

## Prerequisites

- NVIDIA GPU with ≥6GB VRAM (RTX 3050 or better)
- NVIDIA drivers installed (`nvidia-smi` on PATH)
- Git, Python 3.10+
- HuggingFace account + token (for model downloads)

---

## First-Run Sequence

### 1. Install

```powershell
powershell -ExecutionPolicy Bypass -File setup\install_windows.ps1
```

This installs ComfyUI Portable, all custom nodes
(ComfyUI-Manager, Impact-Pack, Impact-Subpack, IPAdapter, ControlNet, GGUF),
creates a Python venv, and downloads all models.

### 2. Generate seed images (no LoRA yet)

```powershell
.venv\Scripts\python scripts\bootstrap_seeds.py --mode closeup
.venv\Scripts\python scripts\bootstrap_seeds.py --mode medium
.venv\Scripts\python scripts\bootstrap_seeds.py --mode fullbody
```

Each run generates 16 candidates. **Pick the best 8** from each batch and copy to:
- `character/seeds/closeup/`
- `character/seeds/medium/`
- `character/seeds/fullbody/`

See `setup/train_lora_guide.md` for curation guidance.

### 3. Prepare training data + train LoRA

```powershell
# Validate 8+8+8 split and image specs
.venv\Scripts\python scripts\prepare_training_data.py --validate

# Generate caption files (review and edit before zipping)
.venv\Scripts\python scripts\prepare_training_data.py --caption-style sdxl

# Package for upload
.venv\Scripts\python scripts\prepare_training_data.py --zip-only
```

Upload `training_data.zip` and `kohya_config.toml` to RunPod or Civitai.
Follow **`setup/train_lora_guide.md`** for the full training walkthrough.

After training, place `KaviB_v1_Prod.safetensors` in:
`C:\ComfyUI\models\loras\`

### 4. Verify the setup

```powershell
.venv\Scripts\python setup\verify_setup.py
```

### 5. Generate images

```powershell
# Single prompt
.venv\Scripts\python scripts\generate.py \
  --prompt "sitting at a sunlit cafe with matcha latte, warm tones" \
  --count 4

# If you hit VRAM issues, use --rescue mode first
.venv\Scripts\python scripts\generate.py --prompt "..." --rescue

# Batch from prompt file
.venv\Scripts\python scripts\batch_generate.py \
  --prompts character\prompts\lifestyle_prompts.txt \
  --count-per-prompt 3 \
  --category lifestyle
```

---

## Key Files

| File | Purpose |
|------|---------|
| `config.yaml` | All paths, defaults, GPU settings |
| `character/character_bible.md` | Persona definition, legal notes, content tiers |
| `character/prompts/base_prompt.txt` | Core prompt fragment (always prepended) |
| `character/prompts/lifestyle_prompts.txt` | Instagram-safe scene prompts |
| `character/prompts/intimate_prompts.txt` | Fanvue/Telegram adult-tier prompts |
| `setup/train_lora_guide.md` | Full LoRA training walkthrough |
| `setup/install_windows.ps1` | One-shot Windows 11 setup |
| `setup/download_models.py` | Download all models via huggingface-hub |
| `setup/verify_setup.py` | End-to-end smoke test |
| `scripts/generate.py` | Main generation CLI |
| `scripts/batch_generate.py` | Batch generation with compliance gate |
| `scripts/bootstrap_seeds.py` | Generate LoRA training seed candidates |
| `scripts/prepare_training_data.py` | Validate, resize, caption, zip dataset |

---

## Troubleshooting

**VRAM OOM:** Add `--rescue` to use the 768×1152 / 24-step safe preset.
If still OOM, add `--no-adetailer`. For FLUX, switch to Q3_K_S GGUF (~4.7GB).

**Character not consistent:** Check that `KaviB_v1_Prod.safetensors` is in
`C:\ComfyUI\models\loras\` and run `verify_setup.py`.

**FaceDetailer missing nodes:** Ensure `ComfyUI-Impact-Subpack` is installed
(separate from Impact-Pack — the subpack contains UltralyticsDetectorProvider).

**401 on model download:** Pass `--hf-token YOUR_TOKEN` and accept model licenses
on HuggingFace before downloading gated models (FLUX, Juggernaut).

See `setup/train_lora_guide.md` for LoRA-specific troubleshooting.
