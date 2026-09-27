# D1 body/framing experiment — learnings (project shelved)

**Status: the AI Influencer project has been shelved by the user.** This
document is a documentation-only record of an in-progress D1 body/framing
investigation at the point work stopped, written at the user's explicit
request to "merge the learnings" before shelving — it is not a resumption
of production. No images were generated, no model was run, and no code,
workflow, prompt, or production YAML was modified to produce this
document. PR #163 (the D1 carousel this investigation was diagnosing)
remains **open and unmerged**, untouched by this work.

## Background

D1's approved concept ("One Blue Dress, Three Moods," PR #163) shipped
with two visible body/framing defects on review: a waist-silhouette
distortion on tight jersey fabric, and full-body slides cropped above the
feet with compressed leg proportions. Kavya authorized bounded,
one-variable-at-a-time testing to diagnose the cause before any
production fix — explicitly not authorizing a production lock without
independent pixel verification. This doc records what those tests found.

## 1. A/A determinism baseline

Ran the exact unchanged production D1 anchor config
(`character/ananya/anchor_libraries/d1_blue_dress_three_moods.yaml`,
`anchor_seed: 334521876`) **twice in direct succession** with zero prompt
or graph changes, via:

```
python scripts/faceswap_carousel.py --anchor-config character/ananya/anchor_libraries/d1_blue_dress_three_moods.yaml --name AATEST_run1 --flux-dev --kontext --anchor-only
python scripts/faceswap_carousel.py --anchor-config character/ananya/anchor_libraries/d1_blue_dress_three_moods.yaml --name AATEST_run2 --flux-dev --kontext --anchor-only
```

Both outputs are **byte-identical**:

```
SHA-256: e99d06f5abab2dab733f2d8d359c1e56b2a34a0b027b1d59065918a05142aa54
```
(`output/2026-09-27/ananya/carousel_AATEST_run1/anchor.png` and
`carousel_AATEST_run2/anchor.png` — local generation output, not
repository-tracked assets)

**Finding, scoped to these two trials:** this pipeline
(`workflows/flux_dev.json`, FLUX dev + fixed seed + `euler`/`simple`
sampler) produced identical output across these two identical-input runs
on this machine — it was deterministic in this pair of trials. This is
not a general claim of pipeline determinism across all conditions (only
two runs were compared, both under identical inputs; no run varying only
hardware/timing/queue state was tested). Within that scope, the
differences seen between the *other* test runs in this investigation
(which had deliberate prompt/graph changes) are more plausibly
attributable to those changes than to run-to-run noise, but that
inference rests on this two-trial result, not on a broader
nondeterminism study.

## 2. Negative-prompt edit: different file hash, zero decoded-pixel difference

`workflows/flux_dev.json` node `"8"` (titled `"Empty Negative"`,
misleadingly — it is not empty) carries a hardcoded anti-thin negative
prompt, sourced as of commit `697578e` (`fix: realism regression +
standardise anchor seed`, #48):

> `skinny, thin, slim, petite, small frame, bony, flat hips, no curves, underweight, stick thin, plastic skin, smooth airbrushed skin, CGI, rendered, artificial studio lighting, overexposed, flat lighting`

This field is not exposed by the anchor YAML schema — nothing in
`scripts/faceswap_carousel.py`'s anchor-config injection path overrides
it. A test run used a private, non-production copy of the graph with only
node `"8"`'s text changed (anti-thin terms removed, quality/lighting terms
kept):

> `plastic skin, smooth airbrushed skin, CGI, rendered, artificial studio lighting, overexposed, flat lighting`

Same seed (334521876), same `EmptySD3LatentImage` 768×1368, same
`KSampler` fields, same body/skin LoRA strengths as production — verified
identical via ComfyUI's own `/history` API at generation time. The
resulting PNG hashed differently from the A/A baseline:

```
SHA-256: 1ce30851ac7f95b5fb4abec4bc3fe295fe020e948230960da7074b06269e099e
```

**But decoded pixel arrays are identical to the A/A baseline — verified
directly** (`numpy` array diff, both images loaded as RGB, `max abs
pixel difference = 0` across all channels).

**Cause, confirmed at two levels:**

1. `workflows/flux_dev.json` node `"7"` (`_claude_inject_seed`, a
   `KSampler`) `inputs.cfg`: value `1.0`.
2. Inspected the actual sampler source shipped with this machine's local
   ComfyUI Desktop install (version `0.22.3`, per
   `comfyui_version.py`, path `comfy/samplers.py` inside the app's
   bundled resources — not a GitHub permalink; the exact upstream commit
   for this pinned version was not independently confirmed against
   `comfyanonymous/ComfyUI` and is not cited as one). In
   `sampling_function()`:

   ```python
   def sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
       if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
           uncond_ = None
       else:
           uncond_ = uncond
       conds = [cond, uncond_]
       ...
   ```

   When `cond_scale` (i.e. the workflow's `cfg` field) is `1.0` and the
   `disable_cfg1_optimization` model option is not set (it is not, in
   this graph), the unconditional/negative branch is set to `None`
   **before** the model's forward pass — the negative prompt's encoded
   conditioning is never evaluated by the model at all for this sampling
   call, not merely algebraically cancelled downstream. This is a named,
   default-on optimization in ComfyUI's own sampler, not a project-
   specific behavior.

**Caveat, stated explicitly per instruction: this finding is scoped to
this graph and this KSampler configuration (`cfg=1.0`), not a universal
claim about negative prompts.** Any workflow using `cfg > 1.0` (e.g. the
anchor generation graph if that field were ever changed, or a different
graph entirely) would not exhibit this — the negative prompt would then
have real effect. The file-hash difference between the two PNGs is
consistent with a different `SaveImage`/execution node id or encoder
metadata difference, not a decoded-pixel difference; this was not traced
further since it wasn't material to the diagnostic question.

## 3. Framing-only wording test — failed

Isolated test: production anchor prompt with only the framing clause
changed, verified via diff to be the only change (body/face/room/outfit
wording held byte-identical):

> OLD: `photorealistic full body editorial fashion photograph,`
> NEW: `standing full-body portrait head to shoes with both feet clearly visible at the bottom of frame, photorealistic editorial fashion photograph,`

Result: feet were **not** included in frame — same crop-above-ankle
defect as the unmodified production anchor. This result was inspected
directly (session-reported and reviewed in-session); it was not
separately re-verified after the fact for this document.

## 4. Composition-clause test — distinct wording, feet now visible; NOT lock-ready

A second, deliberately distinct clause (avoiding "portrait"/"head to
shoes" phrasing) was tested as a private, non-production anchor config
(not committed, not referenced by any carousel prompt file):

> OLD: `photorealistic full body editorial fashion photograph,`
> NEW: `wide-angle full-length shot, camera positioned several steps back to include the entire floor and both feet in frame with visible space below the shoes, photorealistic editorial fashion photograph,`

Same seed (334521876), same LoRA strengths (body 0.5), same face_ref,
same 768×1368 dimensions and sampler config as production — confirmed via
the run's own console output and log at generation time.

```
SHA-256: 4b000d84f4fe2428e46724c9593bb1cf771ea4880d2f7505958a3a6bc43ace83
```
(`.private_test/anchor_composition_v2.png` — **local test evidence, not a
repository asset**)

Result: both bare feet visible, floor/rug included, dress and room intact
— the framing defect this clause targeted did not reproduce.

**Do NOT treat this as lock-ready.** Face/expression drifted from the A/A
baseline, and overall body scale/silhouette changed as well — this was
not a clean single-variable win on visual inspection. The framing fix
appears to work; it does not appear to be free of side effects on other
locked attributes (face, body scale) in this one sample.

**Distinguishing verification levels, per instruction:** the claim that
this test changed *only* the composition clause (one-variable input
equality) was established by diffing the two YAML configs at generation
time and was reported in-session, but has not been independently
re-verified against the current repo state as part of writing this
document — cite it as session-reported, not independently re-confirmed
here. The pixel outcome (feet visible, face/body drift) was independently
pixel-reviewed by Instinct against the rendered PNG. The image was also
sent to Kavya for review; no separate pixel-review findings or approval
were received back in-session — no owner approval or lock should be
inferred from having sent the file.

## 5. Recommended next path, if resumed (not tested, not authorized)

This is a recommendation only — none of it has been tested or authorized
as work to perform:

1. **Face-swap using the existing face reference**
   (`character/ananya/seeds_v2/face_ref_v2.png`) for near-term identity
   consistency across any body/framing fix — this repo's existing ReActor
   faceswap stage already runs this reference against every carousel
   slide as a pipeline stage. This investigation did not test or validate
   its reliability; it's an existing mechanism that would need its own
   verification, not a proven identity lock. The open question from this
   investigation was body/framing, not identity — but "not tested here"
   is not the same as "known to work."
2. **Consider a dedicated persona/body LoRA only after accumulating
   enough curated, consistent images** to train one properly, rather than
   continuing to patch the existing generic body LoRA
   (`Body FIX FLUX.safetensors`) via prompt-only pushes — prompt-only
   pushes were shown in this investigation to interact unpredictably with
   the framing fix (item 4's face/scale drift) rather than composing
   cleanly.

## References

- `character/ananya/anchor_libraries/d1_blue_dress_three_moods.yaml` —
  production D1 anchor (unmodified by this investigation)
- `workflows/flux_dev.json` — anchor generation graph (unmodified;
  negative-prompt field last touched in commit `697578e`, #48)
- `scripts/faceswap_carousel.py` — anchor/slide generation driver
  (unmodified)
- PR #163 — D1 carousel this investigation was diagnosing. **Still open,
  unmerged, untouched by this document or the tests it records.**

## Project status

The AI Influencer project is shelved as of this document. This is a
documentation-only record of the D1 investigation's findings at the point
work stopped — no further production work, image generation, or PR merges
were performed as part of writing it.
