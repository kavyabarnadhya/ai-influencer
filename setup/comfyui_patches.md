# ComfyUI Out-of-Tree Patches

Manual modifications required in your local ComfyUI installation that cannot be
expressed inside this repo (custom_nodes live outside the project tree).
Re-apply after any ComfyUI custom-node update.

---

## 1. ReActor NSFW filter bypass

**Why**: ReActor's bundled NSFW classifier blocks editorial fashion outputs
that include neckline / cleavage descriptors (e.g. deep V-neck slip dresses
on premium-tier slides). When triggered, the node returns a 512×512 black
image instead of the swap result. Premium-tier editorial content is
authorized for Ananya per [CLAUDE.md](../CLAUDE.md).

**File**: `C:\Users\barna\Documents\ComfyUI\custom_nodes\comfyui-reactor\nodes.py`

**Patch** — replace the NSFW-checker block in the swap node's `execute` method
(around lines 452–468, search for the `# NSFW checker` comment):

```python
# Original (blocks editorial content)
# NSFW checker
logger.status("Checking for any unsafe content...")
pbar = progress_bar(len(pil_images))
pil_images_sfw = []
for img in pil_images:
    if state.interrupted or model_management.processing_interrupted():
        logger.status("Interrupted by User")
        break
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    if not sfw.nsfw_image(img_byte_arr, NSFWDET_MODEL_PATH):
        pil_images_sfw.append(img)
    pbar.update(1)
pil_images = pil_images_sfw
# # #
progress_bar_reset(pbar)
```

Replace with:

```python
# NSFW checker bypassed — premium editorial content authorized
pbar = progress_bar(len(pil_images))
pbar.update(len(pil_images))
progress_bar_reset(pbar)
```

**Restart ComfyUI** for the change to load.

**Scope warning**: this disables the NSFW filter globally for every workflow
that uses the ReActor node in your ComfyUI install. If you later add other
characters where editorial premium tier is NOT authorized, restore the
original block and use a separate ComfyUI install for them.

---

## 2. FLUX ControlNet Union Pro v2 (model placement)

**Why**: `workflows/flux_img2img_controlnet.json` references
`flux-controlnet-union-pro-v2.safetensors` via the standard
`ControlNetLoader` node, which reads from `models/controlnet/`.

**Download** (~4 GB):

```bash
curl -L -o "C:/Users/barna/Documents/ComfyUI/models/controlnet/flux-controlnet-union-pro-v2.safetensors" \
  "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0/resolve/main/diffusion_pytorch_model.safetensors?download=true"
```

Restart ComfyUI after placing the file.

---

## Verifying patches

Run `python setup/verify_setup.py` after applying both — it confirms the new
workflow sentinels resolve correctly. Then smoke-test with:

```bash
python scripts/faceswap_carousel.py \
  --prompts character/ananya/carousel_prompts/smoke_test.txt \
  --face-ref character/ananya/seeds_v2/face_ref_v2.png \
  --name verify_patches
```

If smoke_test slide 3 (premium V-neck) returns a black image, the ReActor
patch is not applied or ComfyUI was not restarted after patching.
