# Ananya v2 Seeds

## Directory structure

```
seeds_v2/
  training_canonical/   ← IMMUTABLE after training. Final 25-35 curated images + .txt captions.
  experimental/         ← Work-in-progress generations, faceswap tests, not used in training.

seeds_v2_stock_source/
  licenses/             ← One .txt per image: source URL, license type, download date.
  *.jpg / *.png         ← Original face-obscured stock images (do not modify).

seeds_v2_rejected/
  waxy_skin/            ← Skin texture failed (plastic, over-smoothed).
  semantic_duplicate/   ← CLIP cosine similarity > 0.92 with another accepted image.
  identity_drift/       ← Face diverged from target character.
  over_bokeh/           ← All-bokeh, AI tell, background depth inappropriate.
  uncanny_expression/   ← Uncanny valley expression artifacts.

seeds_v2_failures/      ← Document failure instances for v3 learning.
  bleed/                ← Text encoder LR bleed (background aesthetics in trigger).
  outfit_bake/          ← Outfit absorbed into trigger word.
  overfit/              ← Epoch too late, identity rigid.
  prompt_ignore/        ← Trigger word ignored specific prompt elements.
  hand_corruption/      ← Hand artifacts.

regularization/         ← 30-50 generic Indian woman images. 1 reg per 2 training images.
```

## Rules

- `training_canonical/` is **write-once** after training starts. Never replace or delete files there.
- `AnyV2X9_Prod.safetensors` (once trained) is **immutable baseline**. Any retrain creates a new version — never overwrite Prod.
- Never retrain on files from `experimental/` without full audit first.
- Every file in `seeds_v2_stock_source/` must have a matching license `.txt` in `licenses/`.
- Every rejection goes into the correct subdirectory with a `_reason.txt` alongside the image.
