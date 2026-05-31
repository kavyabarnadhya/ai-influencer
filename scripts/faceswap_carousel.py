"""
Faceswap-driven carousel generator for Ananya v2 (LoRA-free).

Pipeline per slide (2 stages, no compositing):
  1. FLUX schnell img2img off dark anchor body → person+outfit+scene rendered
     together natively. Anchor img2img at moderate denoise locks body proportions
     across slides while prompt fully controls outfit + scene + lighting.
  2. ReActor face swap → identity locked to face_ref_v2.png.

No background compositing, no inpaint, no rembg. FLUX renders person and BG
together in a single forward pass — perspective, scale, and lighting integrate
natively. Body consistency comes from shared anchor seed + img2img denoise.

Two modes:
  - Default (varied outfits): anchor is generic 23yo woman in dark basics.
  - --outfit-lock (OOTD): anchor wears target outfit, slides vary pose/scene only.

Prompt file format:
    denoise=0.75 | <full slide prompt>
    <full slide prompt>    # bare line uses default denoise

See character/ananya/prompt_template.md for prompt structure.

Usage:
    python scripts/faceswap_carousel.py \\
      --prompts character/ananya/carousel_prompts/smoke_test.txt \\
      --face-ref character/ananya/seeds_v2/face_ref_v2.png \\
      --name smoke_08
"""

import copy
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import click
import yaml
from PIL import Image
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from comfyui_api import ComfyUIClient, ComfyUIError, find_comfyui_port, load_workflow, inject_workflow_values
from skin_color_match import match_body_skin_to_face_ref

console = Console()
ROOT = Path(__file__).parent.parent

# Anchor prompt: matches photographic aesthetic of canonical training set.
# Dark-clothed so outfit color in slides doesn't pull toward beige.
# Natural lighting + 50mm framing seeds FLUX with photographic style anchor.
DEFAULT_ANCHOR_PROMPT = (
    "size M hourglass figure woman with fuller bust, slim defined small waist, "
    "balanced curvy hips, slim toned thighs, soft feminine curves, slim with curves "
    "not plus size, South Asian 23 year old, full-body three-quarter standing pose "
    "with dark hair loose over shoulders, plain neutral grey form-fitting bodysuit, "
    "soft natural daylight, plain light grey backdrop, "
    "shot on Kodak Portra 400 film, visible skin pores and faint peach fuzz, "
    "subtle skin imperfections, slight natural oil shine on T-zone, 35mm film grain, "
    "photographic grain noise, candid unretouched amateur photography style, "
    "raw unedited look, no retouching"
)
DEFAULT_DENOISE_VARIED = 0.85
DEFAULT_DENOISE_OUTFIT_LOCK = 0.60


def _inject_flux_t2i(wf: dict, prompt: str, seed: int, propagate_cache: bool = True) -> dict:
    """
    Inject prompt and seed into FLUX T2I workflow.
    Optimization: Uses inject_workflow_values for O(1) node lookup and optimized copying.
    """
    overrides = {
        "_claude_inject_prompt": {"inputs.text": prompt},
        "_claude_inject_seed": {"inputs.seed": seed}
    }
    return inject_workflow_values(wf, overrides, propagate_cache=propagate_cache)


def _inject_flux_img2img(wf: dict, prompt: str, init_image_name: str,
                        denoise: float, seed: int,
                        pose_image_name: str | None = None,
                        cn_strength: float = 0.65,
                        propagate_cache: bool = True) -> dict:
    """
    Inject prompt, init image, denoise, and seed into FLUX img2img workflow.
    Optimization: Uses inject_workflow_values for O(1) node lookup and optimized copying.
    """
    overrides = {
        "_claude_inject_prompt": {"inputs.text": prompt},
        "_claude_inject_init_image": {"inputs.image": init_image_name},
        "_claude_inject_seed": {"inputs.seed": seed, "inputs.denoise": denoise}
    }
    if pose_image_name:
        overrides["_claude_inject_pose_image"] = {"inputs.image": pose_image_name}
        overrides["_claude_inject_controlnet_apply"] = {"inputs.strength": cn_strength}
    return inject_workflow_values(wf, overrides, propagate_cache=propagate_cache)


def _inject_flux_kontext(wf: dict, prompt: str, init_image_name: str, seed: int,
                         bg_lock: bool = True, propagate_cache: bool = True) -> dict:
    """
    Inject prompt, init image, and seed into FLUX Kontext workflow.
    Optimization: Uses inject_workflow_values for O(1) node lookup and optimized copying.
    """
    if bg_lock:
        prompt = f"{prompt}, same background, same scene, unchanged environment"
    overrides = {
        "_claude_inject_prompt": {"inputs.text": prompt},
        "_claude_inject_init_image": {"inputs.image": init_image_name},
        "_claude_inject_seed": {"inputs.seed": seed}
    }
    return inject_workflow_values(wf, overrides, propagate_cache=propagate_cache)


def _inject_faceswap(wf: dict, face_ref_name: str, target_name: str, propagate_cache: bool = True) -> dict:
    """
    Inject face source and target into faceswap workflow.
    Optimization: Uses inject_workflow_values for O(1) node lookup and optimized copying.
    """
    overrides = {
        "_claude_inject_source_image": {"inputs.image": face_ref_name},
        "_claude_inject_target_image": {"inputs.image": target_name}
    }
    return inject_workflow_values(wf, overrides, propagate_cache=propagate_cache)


def _inject_hand_detail(wf: dict, input_image_name: str, seed: int, propagate_cache: bool = True) -> dict:
    """
    Inject input image + seed into FLUX hand detailer workflow (SDXL inpaint on hand bbox).
    """
    overrides = {
        "_claude_inject_input_image": {"inputs.image": input_image_name},
        "_claude_inject_seed": {"inputs.seed": seed}
    }
    return inject_workflow_values(wf, overrides, propagate_cache=propagate_cache)


def _run_and_save(client: ComfyUIClient, wf: dict, out_path: Path,
                  timeout: int = 300) -> Path:
    prompt_id = client.submit_workflow(wf)
    image_refs = client.wait_for_completion(prompt_id, timeout=timeout)
    if not image_refs:
        raise ComfyUIError(f"No output for prompt {prompt_id}")
    img_bytes = client.download_image(
        image_refs[0]["filename"],
        image_refs[0].get("subfolder", ""),
        image_refs[0].get("type", "output"),
    )
    out_path.write_bytes(img_bytes)
    return out_path


def parse_prompts_file(path: Path, default_denoise: float) -> list[dict]:
    """Parse prompt lines. Tokens (any order, all optional):
        denoise=0.85 | anchor=standing | pose=path | cn=0.65 | faceswap=false | <prompt>
    Returns list of dicts: {denoise, anchor, pose, cn_strength, faceswap, prompt}

    faceswap=false skips the ReActor stage for that slide (use for zero-face shots
    — back of head, cropped torso, body-only — where no Ananya face is visible, so a
    swap would only risk distortion). Hand detail + skin lock still run.
    """
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        denoise = default_denoise
        anchor = None
        pose = None
        cn_strength = 0.65
        faceswap = True
        parts = [p.strip() for p in line.split("|")]
        remaining = []
        for part in parts:
            low = part.lower()
            if low.startswith("denoise="):
                try:
                    denoise = float(part.split("=", 1)[1])
                except (ValueError, IndexError):
                    pass
            elif low.startswith("anchor="):
                anchor = part.split("=", 1)[1].strip()
            elif low.startswith("pose="):
                pose = part.split("=", 1)[1].strip()
            elif low.startswith("cn="):
                try:
                    cn_strength = float(part.split("=", 1)[1])
                except (ValueError, IndexError):
                    pass
            elif low.startswith("faceswap="):
                faceswap = part.split("=", 1)[1].strip().lower() not in ("false", "no", "0", "off")
            elif low.startswith("kontext_strength="):
                pass  # FluxKontextImageScale has no strength input — token accepted but ignored
            else:
                remaining.append(part)
        text = " ".join(remaining).strip() if remaining else line
        out.append({"denoise": denoise, "anchor": anchor, "pose": pose,
                    "cn_strength": cn_strength, "faceswap": faceswap, "prompt": text})
    return out


def load_anchor_config(path: Path) -> dict:
    """Load + validate anchor YAML. Two supported schemas:

    SINGLE-ANCHOR (preferred for OOTD — accessory consistency):
        anchor_seed: int (optional)
        anchor_prompt: <full anchor description: body + outfit + accessories + scene>

    MULTI-ANCHOR (legacy — for varied-pose carousels needing back-view / sitting / etc):
        anchor_seed: int (optional)
        shared_tail: str (optional)
        anchors:
          <group_name>:
            prompt: str

    Returns cfg dict with added 'mode' key: 'single' or 'multi'.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: root must be a mapping")

    has_anchors = "anchors" in cfg
    has_single = "anchor_prompt" in cfg
    if has_anchors and has_single:
        raise ValueError(f"{path}: cannot have both 'anchors' (multi mode) and 'anchor_prompt' (single mode)")
    if not has_anchors and not has_single:
        raise ValueError(f"{path}: must have either 'anchor_prompt' (single mode) or 'anchors' (multi mode)")

    if has_single:
        cfg["mode"] = "single"
        if not isinstance(cfg["anchor_prompt"], str) or not cfg["anchor_prompt"].strip():
            raise ValueError(f"{path}: 'anchor_prompt' must be a non-empty string")
    else:
        cfg["mode"] = "multi"
        if not isinstance(cfg["anchors"], dict):
            raise ValueError(f"{path}: 'anchors' must be a mapping")
        if not cfg["anchors"]:
            raise ValueError(f"{path}: 'anchors' must have at least one group")
        for group_name, group_cfg in cfg["anchors"].items():
            if not isinstance(group_cfg, dict) or "prompt" not in group_cfg:
                raise ValueError(f"{path}: anchor group '{group_name}' missing required 'prompt' field")
            if not isinstance(group_cfg["prompt"], str) or not group_cfg["prompt"].strip():
                raise ValueError(f"{path}: anchor group '{group_name}' prompt must be a non-empty string")

    if "anchor_seed" in cfg and not isinstance(cfg["anchor_seed"], int):
        raise ValueError(f"{path}: 'anchor_seed' must be an integer")

    if "anchor_init_denoise" in cfg:
        # Deprecated: img2img body-ref anchor approach was abandoned. Field is ignored.
        v = cfg["anchor_init_denoise"]
        if not isinstance(v, (int, float)) or not (0.0 < float(v) < 1.0):
            raise ValueError(f"{path}: 'anchor_init_denoise' must be float between 0 and 1")

    if "anchor_body_lora_strength" in cfg:
        v = cfg["anchor_body_lora_strength"]
        if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
            raise ValueError(f"{path}: 'anchor_body_lora_strength' must be float between 0.0 and 1.0")

    if cfg["mode"] == "multi":
        _validate_multi_anchor_consistency(cfg, path)

    return cfg


# Outfit/location signal words that must NOT appear exclusively in one anchor group
# (they belong in shared_tail so all anchors render identically).
_OUTFIT_SIGNALS = {
    "fabrics": ["satin", "linen", "silk", "chiffon", "denim", "cotton", "georgette",
                "velvet", "crepe", "organza", "brocade", "khadi"],
    "garments": ["dress", "kurta", "saree", "skirt", "blouse", "top", "shirt",
                 "jumpsuit", "co-ord", "palazzo", "lehenga", "salwar", "dupatta"],
    "colors": ["red", "blue", "green", "yellow", "orange", "pink", "purple", "white",
               "black", "maroon", "mustard", "coral", "rust", "sage", "ivory", "beige",
               "emerald", "navy", "teal", "cream", "olive"],
    "locations": ["balcony", "rooftop", "cafe", "street", "market", "corridor",
                  "garden", "beach", "hotel", "airport", "restaurant", "mall",
                  "delhi", "mumbai", "bandra", "goa", "jaipur", "pondicherry",
                  "chandigarh", "bengaluru", "bangalore"],
}
_ALL_SIGNALS = [w for words in _OUTFIT_SIGNALS.values() for w in words]


def _extract_signals(text: str) -> set[str]:
    lower = text.lower()
    return {w for w in _ALL_SIGNALS if w in lower}


def _validate_multi_anchor_consistency(cfg: dict, path: Path) -> None:
    """Warn if anchor group prompts contain outfit/location signals not in shared_tail.

    Signals in shared_tail = intentional (present in all anchors via concatenation).
    Signals only in a group prompt = risk of per-anchor visual divergence.
    Missing shared_tail entirely = no consistency guarantee → hard error.
    """
    tail = cfg.get("shared_tail", "").strip()
    if not tail:
        raise ValueError(
            f"{path}: multi-anchor config missing 'shared_tail'.\n"
            "  shared_tail carries outfit + accessories + location — required for visual\n"
            "  consistency across anchor groups. Add shared_tail or use single-anchor mode."
        )

    tail_signals = _extract_signals(tail)
    warnings = []
    for group_name, group_cfg in cfg["anchors"].items():
        group_prompt = group_cfg["prompt"].strip()
        group_signals = _extract_signals(group_prompt)
        # Signals in the group prompt but NOT in shared_tail = potential divergence
        exclusive = group_signals - tail_signals
        if exclusive:
            warnings.append(
                f"  anchor '{group_name}' has outfit/location tokens not in shared_tail: "
                + ", ".join(sorted(exclusive))
                + "\n    -> move these to shared_tail or they will diverge across anchor groups"
            )

    if warnings:
        console.print(f"[yellow bold]WARN: Anchor consistency warnings ({path.name}):[/yellow bold]")
        for w in warnings:
            console.print(f"[yellow]{w}[/yellow]")
        console.print(
            "[yellow]  These tokens appear in individual anchor prompts but not shared_tail.\n"
            "  Each anchor is generated independently -- if outfit/location tokens\n"
            "  differ per group, slides will look visually inconsistent.\n"
            "  Recommendation: move all outfit + accessories + location to shared_tail.[/yellow]"
        )


@click.command()
@click.option("--prompts", "prompts_file", default=None, type=click.Path(exists=True))
@click.option("--face-ref", default=None, type=click.Path(exists=True),
              help="Face reference image for ReActor swap. Can also be set via 'face_ref' in --anchor-config YAML.")
@click.option("--name", required=True)
@click.option("--candidates", default=1, show_default=True, type=int)
@click.option("--anchor-seed", default=None, type=int)
@click.option("--outfit-lock", is_flag=True)
@click.option("--anchor-outfit-prompt", default=None)
@click.option("--anchor-prompt", default=None)
@click.option("--anchor-config", default=None, type=click.Path(exists=True),
              help="YAML with anchor(s) — single mode (anchor_prompt:) or multi mode (anchors:)")
@click.option("--flux-dev", is_flag=True,
              help="Use FLUX dev workflows (20 steps, CFG 3.5) instead of schnell (4 steps). ~5x slower, better prompt adherence.")
@click.option("--kontext", is_flag=True,
              help="Use FLUX Kontext Dev for Stage 2 (image editing). Replaces img2img. Requires flux1-kontext-dev-Q4_K_S.gguf.")
@click.option("--character", default="ananya", show_default=True)
@click.option("--anchor-only", is_flag=True,
              help="Generate and save anchor image only — skip all slides. Use to validate body/outfit before a full run.")
def main(prompts_file: str, face_ref: str, name: str, candidates: int,
         anchor_seed: int | None, outfit_lock: bool,
         anchor_outfit_prompt: str | None, anchor_prompt: str | None,
         anchor_config: str | None, flux_dev: bool, kontext: bool, character: str,
         anchor_only: bool):
    """Carousel: FLUX img2img (person+BG together) → ReActor face swap."""

    if not anchor_only and not prompts_file:
        raise click.UsageError("--prompts is required unless --anchor-only is set")

    prompts_path = Path(prompts_file) if prompts_file else None

    # Pick FLUX schnell or dev workflows. Dev is ~5x slower but better prompt adherence.
    flux_wf = {
        "t2i":      "flux_dev.json"                       if flux_dev else "flux_schnell.json",
        "i2i":      "flux_dev_img2img.json"               if flux_dev else "flux_img2img.json",
        "i2i_cn":   "flux_dev_img2img_controlnet.json"    if flux_dev else "flux_img2img_controlnet.json",
    }
    # Per-call timeout: dev img2img+ControlNet runs 10-20min on RTX 3050, schnell ~1-2min
    flux_timeout = 3600 if flux_dev else 300

    # Anchor config mode (single or multi) if --anchor-config provided
    anchor_cfg = None
    if anchor_config:
        anchor_cfg = load_anchor_config(Path(anchor_config))
        # Resolve face_ref from YAML if not passed on CLI
        if face_ref is None:
            face_ref = anchor_cfg.get("face_ref")
        if anchor_seed is None:
            anchor_seed = anchor_cfg.get("anchor_seed", random.randint(1, 2**31 - 1))
        default_denoise = DEFAULT_DENOISE_OUTFIT_LOCK
        if anchor_cfg["mode"] == "single":
            mode_label = "single-anchor (OOTD, accessory-locked)"
        else:
            mode_label = f"multi-anchor ({len(anchor_cfg['anchors'])} groups)"
    elif outfit_lock:
        if not anchor_outfit_prompt:
            console.print("[red]--outfit-lock requires --anchor-outfit-prompt or --anchor-config[/red]")
            raise SystemExit(1)
        anchor_text = anchor_outfit_prompt
        default_denoise = DEFAULT_DENOISE_OUTFIT_LOCK
        mode_label = "outfit-lock (OOTD, single anchor)"
    else:
        anchor_text = anchor_prompt or DEFAULT_ANCHOR_PROMPT
        default_denoise = DEFAULT_DENOISE_VARIED
        mode_label = "varied-outfit"

    if face_ref is None:
        raise click.UsageError("--face-ref is required (or set 'face_ref' in --anchor-config YAML)")
    face_ref_path = Path(face_ref)
    if not face_ref_path.exists():
        raise click.UsageError(f"face_ref not found: {face_ref_path}")

    slides = []
    if not anchor_only:
        slides = parse_prompts_file(prompts_path, default_denoise)
        if not slides:
            console.print(f"[red]No slides parsed from {prompts_path}[/red]")
            raise SystemExit(1)

    if anchor_seed is None:
        anchor_seed = random.randint(1, 2**31 - 1)

    console.print(f"[bold]Faceswap Carousel — {name}[/bold]")
    slide_info = "anchor-only" if anchor_only else str(len(slides))
    console.print(f"Mode: {mode_label} | Slides: {slide_info} | Cands: {candidates}")
    console.print(f"Face ref: {face_ref_path.name} | Anchor seed: {anchor_seed}")
    console.print(f"Default denoise: {default_denoise}")

    port = find_comfyui_port()
    if not port:
        console.print("[red]ComfyUI not running.[/red]")
        raise SystemExit(1)
    console.print(f"ComfyUI on port {port}")
    client = ComfyUIClient(host="127.0.0.1", port=port)

    date_str = datetime.now().strftime("%Y-%m-%d")
    out_dir = ROOT / "output" / date_str / character / f"carousel_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    inter = out_dir / "_intermediate"
    inter.mkdir(exist_ok=True)

    uploaded_face = client.upload_image(str(face_ref_path))

    # Pre-load all workflow templates (needed by both anchor and slide stages)
    t2i_template = load_workflow(str(ROOT / "workflows" / flux_wf["t2i"]))
    i2i_templates = {
        "i2i":     load_workflow(str(ROOT / "workflows" / flux_wf["i2i"])),
        "i2i_cn":  load_workflow(str(ROOT / "workflows" / flux_wf["i2i_cn"])),
        "kontext": load_workflow(str(ROOT / "workflows" / "flux_kontext.json")),
    }

    # Body LoRA strength for anchor t2i — controls _claude_inject_body_lora node in flux_dev.json
    # YAML field anchor_body_lora_strength overrides default; 0.0 = disabled (no-op node)
    _body_lora_strength = float(
        anchor_cfg.get("anchor_body_lora_strength", 0.0)
        if anchor_cfg else 0.0
    )
    if _body_lora_strength > 0.0:
        console.print(f"  Body LoRA strength={_body_lora_strength}")

    # BG lock: append scene-lock token to every Kontext slide prompt.
    # Set bg_lock: false in YAML to disable (e.g. carousels with intentional scene changes).
    _bg_lock = anchor_cfg.get("bg_lock", True) if anchor_cfg else True

    def _make_anchor_wf(prompt_text: str, seed: int) -> dict:
        """Build anchor t2i workflow with body LoRA strength from anchor config."""
        overrides = {
            "_claude_inject_prompt":    {"inputs.text": prompt_text},
            "_claude_inject_seed":      {"inputs.seed": seed},
            "_claude_inject_body_lora": {"inputs.strength_model": _body_lora_strength},
        }
        return inject_workflow_values(t2i_template, overrides)

    # Stage 1: anchor body — single or multi based on config
    uploaded_anchors: dict[str, str] = {}  # group_name -> uploaded filename

    if anchor_cfg and anchor_cfg["mode"] == "single":
        console.print(f"\n[bold cyan]Stage 1 — Single anchor (accessory-locked)[/bold cyan]")
        anchor_prompt_text = anchor_cfg["anchor_prompt"].strip()
        console.print(f"  {anchor_prompt_text[:100]}...")
        anchor_path = inter / "anchor.png"
        wf1 = _make_anchor_wf(anchor_prompt_text, anchor_seed)
        try:
            _run_and_save(client, wf1, anchor_path, timeout=flux_timeout)
            console.print(f"  [green]OK[/green] anchor.png")
        except ComfyUIError as e:
            console.print(f"[red]Anchor failed: {e}[/red]")
            raise SystemExit(1)
        # In single-anchor mode, ALL anchor groups route to this one anchor
        uploaded_single = client.upload_image(str(anchor_path))
        for group in ("default", "standing", "sitting", "closeup", "dynamic"):
            uploaded_anchors[group] = uploaded_single
    elif anchor_cfg and anchor_cfg["mode"] == "multi":
        console.print(f"\n[bold cyan]Stage 1 — Generating {len(anchor_cfg['anchors'])} anchors[/bold cyan]")
        tail = anchor_cfg.get("shared_tail", "").strip()
        for group_name, group_cfg in anchor_cfg["anchors"].items():
            anchor_prompt_text = (group_cfg["prompt"].strip() + " " + tail).strip()
            anchor_path = inter / f"anchor_{group_name}.png"
            console.print(f"  [{group_name}] {anchor_prompt_text[:80]}...")
            wf1 = _make_anchor_wf(anchor_prompt_text, anchor_seed)
            try:
                _run_and_save(client, wf1, anchor_path, timeout=flux_timeout)
                uploaded_anchors[group_name] = client.upload_image(str(anchor_path))
                console.print(f"  [green]OK[/green] anchor_{group_name}.png")
            except ComfyUIError as e:
                console.print(f"[red]Anchor '{group_name}' failed: {e}[/red]")
                raise SystemExit(1)
    else:
        anchor_path = inter / "anchor.png"
        wf1 = _make_anchor_wf(anchor_text, anchor_seed)
        try:
            _run_and_save(client, wf1, anchor_path, timeout=flux_timeout)
            console.print(f"  Saved: {anchor_path.relative_to(ROOT)}")
        except ComfyUIError as e:
            console.print(f"[red]Anchor failed: {e}[/red]")
            raise SystemExit(1)
        uploaded_anchors["default"] = client.upload_image(str(anchor_path))

    if anchor_only:
        anchor_out = out_dir / "anchor.png"
        shutil.copy(inter / "anchor.png", anchor_out)
        console.print(f"\n[bold green]Anchor saved -> {anchor_out.relative_to(ROOT)}[/bold green]")
        console.print("Review body shape and outfit before running full carousel.")
        return

    faceswap_template = load_workflow(str(ROOT / "workflows" / "faceswap_reactor.json"))
    hand_detail_template = load_workflow(str(ROOT / "workflows" / "flux_hand_detail.json"))

    failed = []
    for idx, slide in enumerate(slides):
        denoise = slide["denoise"]
        slide_prompt = slide["prompt"]
        pose_path = slide["pose"]
        cn_strength = slide["cn_strength"]
        anchor_group = slide["anchor"]
        use_cn = pose_path is not None

        # Route to anchor: explicit group, or default fallback
        if anchor_group and anchor_group in uploaded_anchors:
            uploaded_anchor = uploaded_anchors[anchor_group]
            anchor_label = anchor_group
        elif "default" in uploaded_anchors:
            uploaded_anchor = uploaded_anchors["default"]
            anchor_label = "default"
        else:
            # Multi-anchor mode but no anchor= specified and no default → use first
            anchor_label = next(iter(uploaded_anchors))
            uploaded_anchor = uploaded_anchors[anchor_label]
            if anchor_group:
                console.print(f"  [yellow]anchor='{anchor_group}' not in config, using '{anchor_label}'[/yellow]")

        console.print(f"\n[bold]Slide {idx+1}/{len(slides)}[/bold] anchor={anchor_label} denoise={denoise:.2f}"
                      f"{' pose=' + Path(pose_path).name + f' cn={cn_strength:.2f}' if use_cn else ''}")
        console.print(f"  {slide_prompt[:100]}{'...' if len(slide_prompt) > 100 else ''}")

        # Upload pose image once per slide if present
        uploaded_pose = None
        if use_cn:
            pose_full = pose_path if Path(pose_path).is_absolute() else str(ROOT / pose_path)
            if not Path(pose_full).exists():
                console.print(f"  [red]Pose not found: {pose_full}[/red]")
                failed.append((idx, 0, f"Pose not found: {pose_full}"))
                continue
            uploaded_pose = client.upload_image(pose_full)

        for cand in range(candidates):
            slide_seed = random.randint(1, 2**31 - 1)
            base_path = inter / f"slide_{idx:02d}_cand_{cand}_base.png"
            final_path = out_dir / f"slide_{idx:02d}_cand_{cand}.png"

            try:
                # Stage 2: Kontext edit OR standard img2img off anchor
                # Optimization: Skip cache propagation on final injection to avoid extra dict copy in submit_workflow
                if kontext:
                    wf2 = _inject_flux_kontext(
                        i2i_templates["kontext"],
                        slide_prompt, uploaded_anchor, slide_seed,
                        bg_lock=_bg_lock, propagate_cache=False,
                    )
                else:
                    template = i2i_templates["i2i_cn"] if use_cn else i2i_templates["i2i"]
                    wf2 = _inject_flux_img2img(
                        template,
                        slide_prompt, uploaded_anchor, denoise, slide_seed,
                        pose_image_name=uploaded_pose, cn_strength=cn_strength,
                        propagate_cache=False
                    )
                _run_and_save(client, wf2, base_path, timeout=flux_timeout)

                # Stage 3: ReActor face swap — skipped for zero-face slides (faceswap=false).
                # No Ananya face is visible (back of head / cropped torso / body-only), so a
                # swap would only risk distortion. Copy base → final so Stage 3.5/3.6 still run.
                if slide.get("faceswap", True):
                    uploaded_target = client.upload_image(str(base_path))
                    # Optimization: Skip cache propagation on final injection to avoid extra dict copy in submit_workflow
                    wf3 = _inject_faceswap(
                        faceswap_template,
                        uploaded_face, uploaded_target,
                        propagate_cache=False
                    )
                    _run_and_save(client, wf3, final_path, timeout=180)
                else:
                    console.print("  [cyan]faceswap=false — skipping ReActor (zero-face slide)[/cyan]")
                    shutil.copy(base_path, final_path)

                # Stage 3.5: Hand realism — SDXL inpaint on YOLO-detected hand bboxes.
                # Fixes FLUX 6-finger / deformed-hand artefacts. Failures degrade gracefully
                # (ship slide with original FLUX hands rather than abort).
                # ORDER MATTERS: hand detail MUST run BEFORE skin lock (Stage 3.6).
                # Hand inpaint may shift hand skin tone; subsequent skin lock then unifies
                # the whole body skin to face_ref tone. Reversing the order would undo skin lock.
                try:
                    uploaded_for_hands = client.upload_image(str(final_path))
                    wf_hands = _inject_hand_detail(
                        hand_detail_template,
                        uploaded_for_hands,
                        seed=slide_seed,
                        propagate_cache=False,
                    )
                    _run_and_save(client, wf_hands, final_path, timeout=180)
                except Exception as e:
                    console.print(f"  [yellow]hand_detail failed: {e} — shipping uncorrected hands[/yellow]")

                # Skin tone lock: shift body skin to face_ref target (face region untouched).
                # If it fails (missing model, segmentation error, etc.), ship the uncorrected
                # slide rather than abort — uncorrected face/body is still better than no slide.
                try:
                    match_body_skin_to_face_ref(final_path, face_ref_path, final_path)
                except Exception as e:
                    console.print(f"  [yellow]skin_color_match failed: {e} — shipping uncorrected slide[/yellow]")

                # Resize to 1080×1920 (9:16 — Instagram Reels + carousel native res)
                # Original preserved in _intermediate/ base file before overwrite
                img = Image.open(final_path)
                img_resized = img.resize((1080, 1920), Image.LANCZOS)
                img_resized.save(final_path)

                console.print(f"  [green]cand {cand}[/green] -> {final_path.name} (seed {slide_seed})")

            except ComfyUIError as e:
                console.print(f"  [red]cand {cand} FAILED: {e}[/red]")
                failed.append((idx, cand, str(e)))

    console.print(f"\n[bold green]Done.[/bold green] {out_dir.relative_to(ROOT)}")
    if failed:
        console.print(f"[yellow]Failed: {len(failed)}[/yellow]")
        for idx, cand, err in failed:
            console.print(f"  slide {idx} cand {cand}: {err}")


if __name__ == "__main__":
    main()
