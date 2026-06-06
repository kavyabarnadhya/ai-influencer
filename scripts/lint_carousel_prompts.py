"""
Pre-GPU linter for Ananya carousel slide prompt files.

Catches the prompt-class mistakes that have caused real wasted carousel runs
(hands on light fabric -> fused fingers, thin held props on detail shots ->
duplicated glass, head-out shots that still paint a face, S8 forbidden patterns)
*before* any GPU time is spent.

Usage:
    python scripts/lint_carousel_prompts.py character/ananya/carousel_prompts/<name>.txt
    python scripts/lint_carousel_prompts.py <file> --strict   # warnings also fail

Exit codes: 0 = clean (or warnings only without --strict), 1 = errors found.

This is Layer 1 of the carousel QC system (see carousel_workflow.md S16):
  Layer 1 lint (this) -> Layer 2 candidate batching -> Layer 3 mediapipe hand QC.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import click

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- pattern banks -----------------------------------------------------------

# Fabric words that read as LIGHT/PALE -> low contrast -> YOLO hand-detect misses ->
# Stage 3.5 never fires -> fingers fuse into the fabric. Must sit NEXT TO a garment noun
# (so "white pillars" in the BG does not count — only "white ... dress").
_LIGHT = r"(?:white|cream|ivory|beige|pale|light[- ]?(?:grey|gray|blue|pink)|nude|blush|pastel)"
_GARMENT = r"(?:dress|skirt|gown|top|bodice|bodysuit|midi|mini|bodycon|fabric|tank|frock|saree|kurta|lehenga|bandeau)"
_LIGHT_GARMENT = re.compile(
    rf"\b{_LIGHT}\b(?:\s+[\w-]+){{0,4}}\s+\b{_GARMENT}\b"   # white ... dress
    rf"|\b{_GARMENT}\b(?:\s+[\w-]+){{0,4}}\s+\b{_LIGHT}\b",  # dress ... white
    re.I,
)

# A hand resting at/near the waist / hip / midriff / skirt (the repeat deform pose).
_HAND_AT_WAIST = re.compile(
    r"hand[s]?\b[^.|]{0,40}\b(?:at|on|to|near|against|resting on)\b[^.|]{0,20}"
    r"\b(waist|hip|hips|midriff|skirt|thigh|stomach|belly)\b",
    re.I,
)

# Thin held props that FLUX duplicates on tight/detail shots.
_THIN_PROP = re.compile(
    r"\b(champagne\s+flute|wine\s+glass|cocktail\s+glass|flute|wine|cup|straw|cigarette|glass)\b",
    re.I,
)

# Markers that a slide is a tight detail / head-cropped shot.
_DETAIL_SLIDE = re.compile(
    r"\b(detail shot|tight crop|cropped|collarbone down|head (?:fully )?out|head cropped|"
    r"accessory shot|from the collarbone)\b",
    re.I,
)

# A countable object is named (so it needs a singular guard).
_OBJECT_NAMED = re.compile(
    r"\b(bag|handbag|purse|clutch|glass|flute|cup|bouquet|bottle|phone)\b", re.I,
)
_SINGULAR_GUARD = re.compile(r"\b(one single|exactly one|NOT two|NOT duplicate|only one)\b", re.I)

# Negation markers — a clause containing these is a "keep it OUT" instruction, not a pose.
_NEG = re.compile(r"\b(no|not|without|away from|absolutely no)\b", re.I)

# S8 hard-forbidden patterns: (regex, message)
_FORBIDDEN = [
    (re.compile(r"\b(back to camera|body turned away from camera|turned fully away)\b", re.I),
     "180 deg back-to-camera -> Kontext repaints scene, BG collapses. Use 'three-quarter facing toward camera, head over shoulder' (or an intentional faceless walk-away with faceswap=false)."),
    (re.compile(r"hand[s]?\b[^.|]{0,30}\b(?:touch|touching|on|at)\b[^.|]{0,20}"
                r"\b(ribbon|tie|lace|button|zipper|strings?|clasp|knot)\b", re.I),
     "hand on a closure (ribbon/lace/button/zipper) -> Kontext reads as untying the garment. Use hand to cheek / collarbone / in hair."),
    (re.compile(r"\bwaist-?up (?:portrait |)framing\b", re.I),
     "'waist-up framing' is ignored by Kontext (stays full-body). Use 'chest-up portrait framing showing face neck shoulders and neckline only'."),
    (re.compile(r"\bmirror\b", re.I),
     "mirror in BG -> Kontext portal artefact (figure emerging from frame). Use a non-reflective wall/sconce/panel."),
    (re.compile(r"\b(hair flip|hair flung|flinging hair|hair (?:in motion|across (?:the |her )?face))\b", re.I),
     "hair-flip across face -> rubbery artificial strands. Use walking-away or side-profile for hidden face."),
    (re.compile(r"\bboth arms raised straight overhead\b", re.I),
     "straight-overhead arms read stiff/'surrender' and Kontext won't raise them from a relaxed anchor. Bake an armsup: anchor with languid bent elbows."),
    (re.compile(r"\bsitting\b", re.I),
     "'sitting' in a standing carousel -> BG/outfit drift. Sitting = separate post."),
]

# Tokens the parser understands at the start of a slide line.
_TOKEN = re.compile(r"^(anchor|denoise|pose|cn|faceswap|ultra|kontext_strength)=", re.I)


def _strip_tokens(line: str) -> str:
    """Drop leading pipe-tokens, return just the prompt text."""
    parts = [p.strip() for p in line.split("|")]
    kept = [p for p in parts if not _TOKEN.match(p)]
    return " ".join(kept).strip()


def lint_text(text: str) -> tuple[list[str], list[str]]:
    """Lint one full prompt-file string. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    slide_idx = -1
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        slide_idx += 1
        tag = f"slide_{slide_idx:02d}"
        low = line.lower()
        prompt = _strip_tokens(line)
        is_detail = bool(_DETAIL_SLIDE.search(prompt))
        is_light = bool(_LIGHT_GARMENT.search(prompt))
        # Comma-clauses, so a match can be checked against negation in its own clause
        # ("NO hand at the waist" / "NO glass" must NOT trip the positive-pose checks).
        clauses = [c.strip() for c in prompt.split(",")]

        def _positive(rx: re.Pattern) -> str | None:
            """Return the matched text from the first NON-negated clause, else None."""
            for c in clauses:
                m = rx.search(c)
                if m and not _NEG.search(c):
                    return m.group(0)
            return None

        # ERROR: hand at waist/hip on a light-fabric slide -> fused fingers
        if is_light and _positive(_HAND_AT_WAIST):
            errors.append(
                f"{tag}: hand at waist/hip/skirt on a LIGHT fabric -> fingers fuse into the fabric "
                f"(Stage 3.5 misses, ships broken). Move hands to railing / in hair / near face / straight down."
            )

        # ERROR: thin held prop on a detail/head-out slide -> duplicates + claw hands
        prop = _positive(_THIN_PROP) if is_detail else None
        if prop:
            errors.append(
                f"{tag}: thin held prop ({prop}) on a detail/head-out shot -> "
                f"FLUX duplicates the glass and crops hands into claws. Default the detail slide to NO hands + NO prop "
                f"(pure garment/jewelry crop)."
            )

        # WARN: countable object without a singular guard -> may duplicate (precautionary,
        # not always fatal — a held bag usually renders fine; detail shots are the real risk).
        obj = _positive(_OBJECT_NAMED)
        if obj and not _SINGULAR_GUARD.search(prompt) and (is_detail or "holding" in low):
            warnings.append(
                f"{tag}: object '{obj}' named without a singular guard -> FLUX may "
                f"render two. Add 'ONE single ... exactly one NOT two NOT duplicate'."
            )

        # ERROR: head-out / faceless relying on crop but faceswap not disabled is fine; but
        # a head-out detail MUST explicitly exclude the face (FLUX paints one otherwise).
        if is_detail and ("head out" in low or "collarbone down" in low or "head cropped" in low):
            if not re.search(r"no (?:face|chin|head)", low):
                warnings.append(
                    f"{tag}: head-out detail without an explicit 'NO face NO chin' exclusion -> FLUX may paint a "
                    f"(non-Ananya) face. Add 'absolutely NO face NO chin NO neck in frame'."
                )

        # S8 forbidden patterns
        for rx, msg in _FORBIDDEN:
            m = rx.search(prompt)
            if not m:
                continue
            # 'back to camera' / walk-away is OK when paired with faceswap=false
            if "back to camera" in m.group(0).lower() and "faceswap=false" in low:
                continue
            sev = warnings if rx.pattern.startswith(r"\bsitting") else errors
            sev.append(f"{tag}: forbidden pattern '{m.group(0)}' - {msg}")

    return errors, warnings


@click.command()
@click.argument("prompt_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--strict", is_flag=True, help="Treat warnings as failures too.")
def main(prompt_file: Path, strict: bool):
    """Lint a carousel slide prompt file for known GPU-wasting mistakes."""
    errors, warnings = lint_text(prompt_file.read_text(encoding="utf-8"))

    for w in warnings:
        click.echo(f"  WARN  {w}")
    for e in errors:
        click.echo(f"  ERROR {e}")

    n_e, n_w = len(errors), len(warnings)
    if not errors and not warnings:
        click.echo(f"OK  {prompt_file.name}: clean - no known forbidden patterns.")
    else:
        click.echo(f"\n{prompt_file.name}: {n_e} error(s), {n_w} warning(s).")

    if errors or (strict and warnings):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
