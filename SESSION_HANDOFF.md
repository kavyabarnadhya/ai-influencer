# AI Influencer Project — Session Handoff

> **For any AI agent continuing this work:** Read this file fully before touching any code.
> Last updated: 2026-05-13

---

## 1. Project Overview

A local, CLI-driven pipeline to generate face-consistent images of an AI influencer named **Ananya** using ComfyUI + SDXL on a Windows 11 machine with an RTX 3050 (6GB VRAM).

**Stack:**
- `ComfyUI Desktop` — local inference server (http://localhost:8000)
- `Juggernaut XL v9` — primary SDXL checkpoint
- `AnanyaAI_v1_Prod.safetensors` — Ananya character LoRA (SDXL, production)
- `flux1-schnell-Q4_K_S.gguf` — FLUX model (Q4 quantised, preferred; Q3 fallback)
- `Ollama` (model: `dolphin-llama3`) — uncensored LLM for prompt polishing
- Python `.venv` at project root (`venv/Scripts/python.exe`); scripts under `scripts/`
- Branch: `feature/nl-prompt-assistant` (active, not merged to main)

**Content strategy:** Bollywood-adjacent fashion, Instagram-first. Free tier: fully clothed. Subscription/premium tier: cleavage-visible, editorial fashion. NOT explicit/adult content.

---

## 2. Current State

### Production (working now)
```powershell
# Single image
.venv\Scripts\python.exe scripts\generate.py --prompt "rooftop portrait, golden hour"

# Carousel (best current pipeline)
.venv\Scripts\python.exe scripts\generate_carousel.py `
  --scene "soft dusk rooftop, sharp city background, warm ambient terrace lights, Mumbai skyline" `
  --outfit "berry pink fitted bodycon dress, scoop neck, short sleeves, above-knee hem" `
  --hair "loose waves, side parted, natural" `
  --slides 4 --name my_carousel `
  --face-ref "character/ananya/reference_board/face_ref_001_2890463320.png" `
  --poses-dir "character/ananya/poses/carousel_production_v2" `
  --candidates 3

# NL prompt assistant
.venv\Scripts\python.exe scripts\prompt_assistant.py "rooftop portrait golden hour"
```

### Ananya v2 LoRA — Dataset Complete, Captioning Next

**33 images** in `character/ananya/seeds_v2/training_canonical/` — fully curated, diversity verified.

**Diversity confirmed (visual audit 2026-05-13):**
- Shot types: extreme closeup, closeup, medium, fullbody, back-turned, seated
- Outfits: saree (7), lehenga (7), salwar kameez (2), Western casual/fashion (12)
- Lighting: outdoor day, golden hour, night/fairy lights, studio neutral, dramatic coloured, indoor cafe
- Expressions: neutral, looking away, smile, eyes closed, back to camera
- Premium images included intentionally: deep V neckline, open jacket — subscription tier training

**Bootstrap images REJECTED** (`experimental/bootstrap_2026-05-09_*/`) — IPAdapter caused skin tone drift toward lighter/Southeast Asian features. Do not add to training_canonical.

---

## 3. NEXT SESSION — Caption the Dataset

### Step 1: Run Florence-2 auto-caption (~30 min, CPU)
```powershell
.venv\Scripts\python.exe scripts\auto_caption.py `
  --input-dir "character/ananya/seeds_v2/training_canonical" `
  --mode florence2
```
- Writes one `.txt` draft per image, prepended with trigger `AnyV2X9`
- Florence-2 downloads automatically on first run (~900MB)

### Step 2: Manually edit all 33 `.txt` files
**Caption rules (strict):**
1. `AnyV2X9` must be FIRST token — always
2. **OMIT:** face shape, eye color, skin tone, ethnicity, body type, age, hair color
3. **INCLUDE:** shot type, focal length feel, camera angle, hair style+state, outfit details, jewelry, pose, expression, lighting, DOF, aesthetic, geographic anchor
4. Use exact vocabulary from `character/ananya/v2_scene_anchor_vocab.md`

**Caption template:**
```
AnyV2X9, {shot_type}, {focal_phrase} seen from {camera_angle} at {elevation},
with {hair_style} {hair_state}. She is {pose_action} and expressing {emotion}.
{lighting_phrase}, {dof_phrase}, {aesthetic_mode_phrase}, {geographic_anchor_phrase}.
```

### Step 3: CLIP similarity audit
```powershell
.venv\Scripts\python.exe scripts\clip_similarity_audit.py `
  --input-dir "character/ananya/seeds_v2/training_canonical"
```
Reject any image scoring < 0.2 cosine similarity to the mean (outlier/bad swap).

### Step 4: Zip + RunPod training
```powershell
.venv\Scripts\python.exe scripts\prepare_training_data.py --character ananya --zip-only
```
- Upload `training_data_ananya.zip` + `setup/kohya_config.toml` to RunPod
- Use `ostris/ai-toolkit` for FLUX Dev LoRA training
- GPU: RTX A6000 48GB @ ~$0.49/hr, ~1.5-3hr training
- Output: `AnanyaAI_v2_Prod.safetensors` → `Documents\ComfyUI\models\loras\`

---

## 4. Other Pending Work

| Task | Priority | Notes |
|------|----------|-------|
| Test `carousel_production_v3` poses | Medium | 6 new poses in `character/ananya/poses/carousel_production_v3/`, untracked, untested |
| Merge `feature/nl-prompt-assistant` → main | Low | Stable, 15+ commits ahead |
| Generate real content carousels | Medium | v1 LoRA production-ready now |

---

## 5. Key Files

| File | Purpose |
|------|---------|
| `config.yaml` | Central config: models, paths, generation settings |
| `scripts/generate_carousel.py` | Main carousel pipeline |
| `scripts/generate.py` | Single image generation |
| `scripts/prompt_assistant.py` | NL → prompt via Ollama |
| `scripts/faceswap_stock.py` | Batch faceswap — has `--files` flag for selective runs |
| `scripts/auto_caption.py` | Auto-caption training images (florence2 or stub mode) |
| `scripts/clip_similarity_audit.py` | CLIP outlier detection on training set |
| `character/ananya/v2_scene_anchor_vocab.md` | Exact vocabulary for v2 captions |
| `character/ananya/seeds_v2/training_canonical/` | **33 curated training images** |
| `workflows/t2i_sdxl_lora_ipadapter_controlnet.json` | Best carousel workflow |

---

## 6. Hardware

| Item | Value |
|------|-------|
| OS | Windows 11 |
| GPU | NVIDIA RTX 3050 6GB |
| ComfyUI | `C:\Users\barna\Documents\ComfyUI` (port 8000) |
| Project | `C:\Users\barna\Projects\ai-influencer` |
| FLUX model | `flux1-schnell-Q4_K_S.gguf` (preferred), Q3 fallback |
| SDXL checkpoint | `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors` |
| Ollama | `dolphin-llama3` |
