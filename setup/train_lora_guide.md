# Cloud LoRA Training Guide — KaviB Character

Train a Kohya Dreambooth LoRA on a cloud GPU (RunPod or Civitai) — one-time cost ~$2–8.
The result is a single `KaviB_v1_Prod.safetensors` file (~145MB) that you drop into ComfyUI
and use for every generation going forward.

---

## Phase 1 — Dataset Curation (24 images, 1024×1024 PNG)

### Required split

| Category | Count | Framing |
|----------|-------|---------|
| `character/seeds/closeup/` | 8 | Chin to forehead — emphasize skin pores and iris |
| `character/seeds/medium/` | 8 | Waist up — capture shoulder width and hair silhouette |
| `character/seeds/fullbody/` | 8 | Head to toe — full figure in context of background |

### Generation (before you have the LoRA)

Use `scripts/bootstrap_seeds.py` to generate candidates with Juggernaut XL (no LoRA yet):

```powershell
.venv\Scripts\python scripts\bootstrap_seeds.py --mode closeup   # generates 16 candidates
.venv\Scripts\python scripts\bootstrap_seeds.py --mode medium    # generates 16 candidates
.venv\Scripts\python scripts\bootstrap_seeds.py --mode fullbody  # generates 16 candidates
```

From each batch, manually pick the **best 8** and copy them to the appropriate `seeds/` folder.
Curation criteria: sharpest face, most varied poses/lighting, no visible SDXL artifacts.

### Supplementing with real photos (strongly recommended)

Training from only 24 synthetic images risks "teaching back" SDXL rendering artifacts instead
of learning a robust identity. Supplement with **6–8 real-camera reference photos**:

- Own photos of a consenting adult model with explicit permission, or
- Properly licensed stock photos of a consenting adult model.
- **Never** celebrities, influencers, or scraped/third-party faces.

Place these in the appropriate `seeds/` subfolders. The validator counts them in the totals.

### Image standards

- Format: **PNG** (not JPEG, not WebP)
- Size: **1024×1024** pixels exactly (pixel dimensions matter; DPI metadata is irrelevant)
- Color mode: RGB

`prepare_training_data.py` auto-converts and resizes non-conforming images.

---

## Phase 2 — Captioning (Isolation Rule)

**Trigger word: `KaviB`** — always the first token, keep_tokens = 1.

Caption rule: describe **everything except** permanent identity features (face shape, bone
structure, hair color/length, eye shape, skin tone). The model learns:
*"KaviB is the constant; everything else is a variable."*

| Caption mode | When to use | Rule |
|-------------|-------------|------|
| SDXL (default) | Training on Juggernaut XL | WD14-style tags, prepend `KaviB, `, strip face/hair tags |
| FLUX | Training on FLUX base model | JoyCaption natural language, prepend `A photo of KaviB, `, omit face descriptors |

Tags to strip (SDXL mode): `brown eyes`, `short hair`, `black hair`, `asian`, `sharp eyes`,
`detailed face`, `beautiful face`, `pretty face`, `young woman`, `girl`, `1girl`.

Generate captions automatically:

```powershell
.venv\Scripts\python scripts\prepare_training_data.py --caption-style sdxl
```

Then **review and edit** every `.txt` file in `character/seeds/**/*.txt`. Remove any remaining
face/identity descriptors. Captions should describe only: clothing, setting, pose, lighting, mood.

Example good caption:
```
KaviB, sitting at an outdoor cafe, wearing a white linen blouse, warm golden hour lighting,
bokeh background, potted plants, coffee cup on table, relaxed pose
```

Example bad caption (contains identity descriptors — remove these):
```
KaviB, short haired young asian woman, brown eyes, beautiful face, ...
```

---

## Phase 3 — Kohya SS Config (copy-paste ready TOML)

```toml
[general]
enable_bucket = true
bucket_reso_steps = 64
min_bucket_reso = 256
max_bucket_reso = 1024

[datasets.[[datasets]].subsets]
image_dir = "./train_data"
caption_extension = ".txt"
num_repeats = 10
keep_tokens = 1

[model_arguments]
pretrained_model_name_or_path = "RunDiffusion/Juggernaut-X-v10"
v2 = false

[network_arguments]
network_module = "networks.lora"
network_dim = 32
network_alpha = 32

[optimizer_arguments]
optimizer_type = "Prodigy"
optimizer_args = ["d_coef=2", "use_bias_correction=True", "safeguard_warmup=True"]
learning_rate = 1e-4

[training_arguments]
output_dir = "./output"
output_name = "KaviB_v1_Prod"
save_model_as = "safetensors"
max_train_steps = 2400
noise_offset = 0.0357
min_snr_gamma = 5.0
mixed_precision = "fp16"
xformers = true
cache_latents = true
cache_latents_to_disk = true
gradient_checkpointing = true

[sample_prompts]
sample_every_n_steps = 200
sample_sampler = "dpmpp_2m"
sample_prompts = [
  "a photo of KaviB standing in a crowded city street, evening lighting",
  "a close up of KaviB laughing, wearing a yellow hat",
  "KaviB sitting on a park bench, full body shot"
]
```

`prepare_training_data.py --zip-only` writes this TOML automatically as `kohya_config.toml`.

### Key parameter notes

| Parameter | Value | Why |
|-----------|-------|-----|
| `optimizer_type` | Prodigy | Auto-adjusts learning rate — eliminates manual LR tuning |
| `network_dim` / `network_alpha` | 32 / 32 | Balanced capacity; file size ~145MB |
| `max_train_steps` | 2400 | 24 images × 10 repeats × ~10 steps each |
| `noise_offset` | 0.0357 | Prevents flat/overexposed lighting in SDXL |
| `min_snr_gamma` | 5.0 | Balances noise schedule for better skin texture |
| `sample_every_n_steps` | 200 | Visual checkpoint every ~200 steps — pick best visually |

---

## Phase 4 — Cloud Platform Steps

### RunPod (preferred — full control)

1. Go to https://runpod.io → Pods → + New Pod
2. Select a template: **Kohya_ss** or **AI-Toolkit** (search template library)
3. Choose GPU: RTX 3090 or 4090 (fastest for SDXL training)
4. Upload your files to the pod:
   - `training_data.zip` (from `prepare_training_data.py --zip-only`)
   - `kohya_config.toml`
5. In the pod terminal:
   ```bash
   unzip training_data.zip -d train_data
   accelerate launch train_network.py --config_file kohya_config.toml
   ```
6. Watch `/output/sample/` — images appear every 200 steps (steps 200, 400, 600…)
7. Evaluate each sample batch: face sharpness + flexibility (can it handle new scenes?)
8. Stop training when samples look consistent but not over-specific
9. Download the best checkpoint: `KaviB_v1_Prod.safetensors` (should be < 200MB)

### Civitai On-Site Trainer (simpler — less control)

1. Go to https://civitai.com/models/train
2. Upload your 24 images (the processed PNGs from `training_data/`)
3. Set: trigger word `KaviB`, rank 32, optimizer Prodigy
4. Start training; Civitai trains in the background
5. Download the `.safetensors` file when complete

---

## Phase 5 — Deploy

1. Rename the downloaded file to `KaviB_v1_Prod.safetensors` if needed
2. Place it at: `C:\ComfyUI\models\loras\KaviB_v1_Prod.safetensors`
3. Verify it loads:
   ```powershell
   .venv\Scripts\python setup\verify_setup.py
   ```
4. Generate a test image:
   ```powershell
   .venv\Scripts\python scripts\generate.py --prompt "standing in a park, sunny day" --count 2
   ```

Check that the character is consistent across both images and matches your seed dataset.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Face looks generic / no KaviB identity | LoRA not loading | Check `config.yaml` lora path; run `verify_setup.py` |
| Face is "plastic" or distorted | Double-baked anatomy | Remove face descriptors from prompts — only use `KaviB` trigger |
| Character drifts across images | LoRA strength too low | Increase `lora_strength` in config.yaml (try 0.9) |
| Overfit — character in every scene regardless of prompt | Trained too long | Use an earlier checkpoint (step 1200–1800 range) |
| LoRA file > 200MB | Rank too high | Stick to rank 32; absolute max rank 64 |
| FLUX workflow ignores KaviB identity | SDXL LoRA ≠ FLUX UNet | Expected — FLUX is text-only until you train a separate FLUX LoRA |
