# AI Influencer Project — Session Handoff

> **For any AI agent continuing this work:** Read this file fully before touching any code.
> Last updated: 2026-04-27

---

## 1. Project Overview

A local, CLI-driven pipeline to generate face-consistent images of an AI influencer
named **Ananya** using ComfyUI + SDXL on a Windows 11 machine with an RTX 3050 (6GB VRAM).

**Stack:**
- `ComfyUI` — local inference server (http://localhost:8188)
- `Juggernaut XL v9` — primary SDXL checkpoint
- `AnanyaAI_v1_Prod.safetensors` — Ananya character LoRA (SDXL)
- `flux1-schnell-Q3_K_S.gguf` — FLUX model (quantised for 6GB VRAM)
- `Ollama` (model: `dolphin-llama3`) — uncensored LLM for prompt polishing
- Python `.venv` at project root; scripts under `scripts/`

**Key commands:**
```powershell
# Start ComfyUI (must be running before generating)
cd C:\Users\barna\Documents\ComfyUI
.\.venv\Scripts\python.exe main.py --listen --lowvram

# Generate image (fast, best Ananya identity)
cd C:\Users\barna\Projects\ai-influencer
.\.venv\Scripts\python.exe scripts\prompt_assistant.py "scene description here" --rescue

# Generate with full quality (no --rescue, needs GPU free of other apps)
.\.venv\Scripts\python.exe scripts\prompt_assistant.py "scene description here"

# Generate with FLUX background (experimental, has identity loss issue — see Section 3)
.\.venv\Scripts\python.exe scripts\prompt_assistant.py "scene description" --use-flux-bg --rescue

# Review mode — edit prompt before generating
.\.venv\Scripts\python.exe scripts\prompt_assistant.py "scene" --review

# Dry run — print polished prompt only
.\.venv\Scripts\python.exe scripts\prompt_assistant.py "scene" --dry-run
```

---

## 2. Current State of the Codebase

All changes committed on branch: `feature/nl-prompt-assistant`

| File | Purpose |
|------|---------|
| `config.yaml` | Central config: models, paths, generation settings, rescue mode |
| `scripts/prompt_assistant.py` | Main CLI: NL→prompt, optional FLUX 2-pass orchestration |
| `scripts/generate.py` | Core generation: loads workflow, injects params, calls ComfyUI API |
| `scripts/comfyui_api.py` | ComfyUI HTTP client |
| `scripts/mcp_server.py` | MCP server wrapper (for Claude Desktop / agent use) |
| `workflows/t2i_sdxl_lora.json` | **Best workflow** — pure SDXL + LoRA + FaceDetailer + HandDetailer |
| `workflows/t2i_sdxl_lora_backup.json` | Backup of the above (do not modify) |
| `workflows/t2i_img2img.json` | Img2Img workflow for FLUX 2-pass (experimental) |
| `workflows/t2i_ipadapter.json` | Old IP-Adapter workflow (kept for reference) |
| `workflows/flux_schnell.json` | FLUX Schnell background generation workflow |
| `character/ananya/base_prompt.txt` | Physical description tags only (no scene/style bias) |
| `character/ananya/reference_board/` | Ground-truth training images used for the LoRA |

---

## 3. Known Issues and Their Status

### ✅ FIXED: FLUX CFG settings were wrong
FLUX Schnell was being called with SDXL settings (30 steps, 8.0 CFG), producing fried
images. Now auto-corrected in `generate.py`: FLUX always uses 4 steps + 1.0 CFG.

### ✅ FIXED: Rescue mode didn't use LoRA
`rescue_mode.workflow` was pointing to `bootstrap_seeds` (no LoRA). Now points to
`t2i_sdxl_lora`, giving Ananya her correct identity even in low-VRAM mode.

### ✅ FIXED: Prompt contradictions (outdoor light in nightclub)
`base_prompt.txt` had "candid street style photography" baked in.
LLM rules now separate indoor vs. outdoor lighting explicitly.

### ⚠️ KNOWN LIMITATION: FLUX 2-pass identity loss
The `--use-flux-bg` flag is experimental. The pipeline:
1. Generates a FLUX background (with a random woman in it).
2. Uses Img2Img (denoise 0.65) to paint Ananya over it.

**Problem:** Because FLUX always generates a different random woman, the underlying
bone structure bleeds through into the final face despite the LoRA. There is no
reliable fix for this without ControlNet (needs 12GB+ VRAM).

**Workaround:** Do NOT use `--use-flux-bg` for final content. Use plain SDXL.
The SDXL output with `t2i_sdxl_lora.json` gives the **best Ananya identity match**.

---

## 4. NEXT SESSION: Train FLUX LoRA on RunPod

This is the highest-priority next task. It will permanently solve the identity problem
by giving FLUX native knowledge of Ananya's face.

### Why
A FLUX LoRA trained on the Ananya reference board will let us drop `--use-flux-bg`
entirely. A single FLUX Schnell pass (4 steps, ~60 seconds on RTX 3050) will produce
photorealistic backgrounds AND a perfect Ananya face in one go.

### Training Rules (Critical)
- **ALWAYS train on FLUX.1 [Dev]** — never Schnell (Schnell is distilled, breaks training)
- **LoRAs trained on Dev run perfectly on Schnell** — this is the standard workflow
- Training tool: **`ostris/ai-toolkit`** (the current gold standard for FLUX LoRA training)
- Recommended GPU on RunPod: **RTX A6000 (48GB VRAM)** @ ~$0.49/hr
- Estimated training time: **1.5 to 3 hours** (well within the $10 budget)
- Output file size: approximately 150MB to 500MB

### Training Data
Reference images already prepared at:
`C:\Users\barna\Projects\ai-influencer\character\ananya\reference_board\`
- `primary_ref_200138.png` — primary identity reference
- `face_ref_001_2890463320.png` — secondary close-up

The user should prepare **15–30 cropped, clean face images** before the RunPod session.
Diverse lighting and angles are important for a robust LoRA.

### Step-by-Step RunPod Plan
1. Go to https://www.runpod.io/console/pods
2. Deploy a Pod: **RTX A6000 (48GB VRAM)** @ $0.49/hr
3. Template: **RunPod PyTorch 2** official template
4. Container Disk: 50 GB | Volume Disk: 100 GB
5. Connect to **JupyterLab** once the pod is Running
6. Run these commands in the terminal:
```bash
# Install ai-toolkit
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
pip install -r requirements.txt

# Download FLUX Dev model (this will take ~15 minutes)
huggingface-cli download black-forest-labs/FLUX.1-dev \
  flux1-dev.safetensors --local-dir ./models/flux

# Upload your training images to /workspace/training_images/
# (use JupyterLab file browser to drag and drop)

# Then configure config/examples/train_lora_flux_24gb.yaml
# and run training:
python run.py config/examples/train_lora_flux_24gb.yaml
```
7. Download the output `.safetensors` file when training completes.
8. Place it in: `C:\Users\barna\Documents\ComfyUI\models\loras\AnanyaAI_FLUX_v1.safetensors`

### After Training: Update the FLUX Workflow
- The existing `workflows/flux_schnell.json` needs a `LoraLoader` node added
- The `config.yaml` needs a new `lora` key under the `ananya` character pointing to the FLUX LoRA
- `generate.py` needs a check: if workflow is `flux_schnell`, inject the FLUX LoRA (not the SDXL LoRA)

---

## 5. Hardware Context

| Item | Value |
|------|-------|
| OS | Windows 11 |
| GPU | NVIDIA RTX 3050 6GB |
| Storage constraint | C: drive near full — avoid large downloads |
| ComfyUI location | `C:\Users\barna\Documents\ComfyUI` |
| Project location | `C:\Users\barna\Projects\ai-influencer` |
| FLUX model | `flux1-schnell-Q3_K_S.gguf` (Q3 quantised, ~4GB) |
| SDXL checkpoint | `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors` |
| Ollama model | `dolphin-llama3` (uncensored, needed for fashion prompts) |
