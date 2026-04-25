# KaviB LoRA Training Guide

## Overview

Train a Dreambooth LoRA using Kohya SS on RunPod (~$2–8 one-time cost). The LoRA encodes KaviB's face identity so every generation is face-consistent without needing InsightFace/InstantID.

## Step 1 — Prepare Training Dataset

**Generate seed candidates:**
```powershell
python scripts\bootstrap_seeds.py --mode closeup --count 16
python scripts\bootstrap_seeds.py --mode medium --count 16
python scripts\bootstrap_seeds.py --mode fullbody --count 16
```

**Curate:**
- Open `character/seeds/{closeup,medium,fullbody}/`
- Keep only the **8 best** from each mode (most consistent face, no artifacts)
- Delete rejects

**Validate:**
```powershell
python scripts\prepare_training_data.py --validate
```

## Step 2 — Caption the Images

```powershell
python scripts\prepare_training_data.py --caption-style sdxl
```

This generates a `.txt` file for every image. **Review and edit every caption manually.**

**Isolation Rule — caption ONLY what changes, not identity:**

| ✅ Include in caption | ❌ Never include |
|-----------------------|-----------------|
| Clothing, outfit style | Face shape, eye color |
| Setting, location | Hair color, hair length |
| Pose, body position | Skin tone, ethnicity |
| Lighting, mood | "woman", "person", "KaviB" (trigger is prepended automatically) |
| Camera angle | Any permanent physical feature |

**Example caption:** `wearing a white linen blouse, standing in a sunny park, waist-up shot, relaxed pose, warm natural light`

**Bad caption:** `short-haired south asian woman with dark eyes wearing a white blouse` ← double-baking!

## Step 3 — Package for Upload

```powershell
python scripts\prepare_training_data.py --zip-only
```

This produces `training_data.zip` containing:
- `img/10_KaviB woman/` with all 24 PNG + TXT pairs
- `setup/kohya_config.toml`

## Step 4 — Train on RunPod

1. Create a RunPod account: https://runpod.io
2. Launch a **RTX 3090 or A100** pod with the **Kohya SS** template
3. Upload `training_data.zip` to the pod
4. Edit `kohya_config.toml` — fill in `YOUR_*` placeholders with actual paths
5. Run training:
   ```bash
   accelerate launch train_network.py --config_file kohya_config.toml
   ```
6. Monitor loss: target 0.05–0.12 range; stop early if loss spikes
7. Training produces checkpoint files every 200 steps — pick the best one (usually 1600–2400 steps)

**Estimated cost:** $2–8 depending on GPU tier and training duration (~30–60 min)

## Step 5 — Deploy the LoRA

1. Download `KaviB_v1_Prod.safetensors` from RunPod
2. Copy to: `C:\ComfyUI\models\loras\KaviB_v1_Prod.safetensors`
3. Verify:
   ```powershell
   python setup\verify_setup.py
   ```
4. Test generation:
   ```powershell
   python scripts\generate.py --prompt "sitting at a cafe with matcha latte"
   ```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Plastic/distorted face | Caption has anatomy — remove all physical descriptors |
| Face not consistent across images | Increase LoRA strength in `config.yaml` (`lora_strength: 0.9`) |
| Loss NaN after first step | Reduce learning rate; check for corrupt training images |
| OOM during training | Reduce `train_batch_size` to 1, enable `gradient_checkpointing` |
| Face too strong / cartoonish | Reduce LoRA strength to 0.7–0.75 |
