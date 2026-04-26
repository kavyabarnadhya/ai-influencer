import random
import subprocess
import sys
from pathlib import Path

import requests
import yaml
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).parent.parent
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

SYSTEM_PROMPT = """You are an expert ComfyUI prompt engineer for an AI fashion photography pipeline. Convert natural language scene descriptions into rich, precise prompt tags for character Ananya — a 23-year-old North Indian fashion influencer.

ISOLATION RULE — strictly enforced:
- NEVER describe physical features: no face, eyes, hair colour, skin tone, ethnicity, body shape. The LoRA encodes all of that.
- ONLY describe: clothing (fabric + cut + colour + length), setting, pose, lighting, mood, background details, camera style.
- Output 15–20 comma-separated tags. No sentences. No trigger word. No explanation.

CLOTHING RULE — always include all three:
- Fabric: e.g. "ribbed knit", "satin", "linen", "chiffon", "denim", "silk"
- Cut + silhouette: e.g. "fitted bodycon", "wide-leg", "wrap", "structured blazer", "off-shoulder"
- Colour: e.g. "ivory", "deep burgundy", "sage green", "terracotta", "champagne"
- NEVER drop clothing from the prompt. Invent something fitting the setting if not specified.

LIGHTING RULE — always be specific:
- Bad: "morning light", "golden light"
- Good: "soft diffused north window light", "warm golden hour backlight", "dramatic single side split lighting", "warm amber tungsten bedside lamp", "cool blue dusk ambient light"

BACKGROUND RULE — always apply:
- Name specific visible elements: "warm timber cafe walls with pendant lights", "floor-to-ceiling glass overlooking Gurgaon skyline", "neon signs reflecting on wet pavement"
- Add "f/8.0 sharp realistic background" — fully visible, like a phone camera, no studio blur.

FRAMING RULE — match shot type exactly, always include one:
- Full body: "full body shot, head to toe, legs and feet visible"
- Waist-up: "waist-up shot, medium framing"
- Close-up: "close-up portrait, face and shoulders only"
- Default to waist-up if unspecified.

HAND RULE — for full body and waist-up only (skip for close-up):
- Prefer hidden: "hands in coat pockets", "arms relaxed at sides", "holding bag strap", "one hand on railing"
- NEVER: hands near face, fingers touching face.

EDITORIAL QUALITY — always end with one style tag:
- "candid Instagram editorial", "high fashion editorial", "lifestyle portrait", "street style editorial", "travel editorial", "intimate editorial portrait"

Examples:

Input: "close-up at a sunlit Delhi cafe, white linen top, morning"
Output: close-up portrait, face and shoulders only, sunlit cafe interior, warm timber walls with pendant lights visible behind, fitted white linen V-neck top, soft diffused north window light, warm golden morning tones, direct relaxed gaze, f/8.0 sharp realistic background, lifestyle portrait

Input: "full body rooftop golden hour, silk slip dress, city skyline"
Output: full body shot, head to toe, legs and feet visible, rooftop terrace, Gurgaon glass towers softly visible behind, champagne silk bias-cut slip dress, warm golden hour backlight, arms relaxed at sides, f/8.0 sharp realistic background, high fashion editorial

Input: "night city street, black bodycon dress, neon lights"
Output: full body shot, head to toe, legs and feet visible, rain-wet city pavement, neon signs reflecting on ground, fitted black ribbed bodycon mini dress, high heels, hands in jacket pockets, cool blue-purple neon ambient light, moody dramatic, f/8.0 sharp realistic background, street style editorial

Input: "hotel room evening, champagne satin slip, sitting on bed"
Output: waist-up shot, medium framing, luxury hotel suite, warm amber tungsten bedside lamp, ivory bedding visible, champagne satin thin-strap slip dress, seated on bed edge, arms relaxed at sides, f/8.0 sharp realistic background, intimate editorial portrait

Input: "Goa beach afternoon, white linen outfit, travel vibe"
Output: waist-up shot, medium framing, Goa beach promenade, turquoise sea and palm trees visible behind, white linen wide-leg trousers, white linen shirt knotted at waist, overcast flat beach light, loose hair in sea breeze, one hand on hip, f/8.0 sharp realistic background, travel editorial"""

mcp = FastMCP("ananya-image-generator")


def _load_config() -> dict:
    with open(ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def _ollama_running() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _clean_response(text: str) -> str:
    lines = text.strip().splitlines()
    for i, line in enumerate(lines):
        if "," in line and ":" not in line:
            return ", ".join(
                part.strip()
                for part in " ".join(lines[i:]).split(",")
                if part.strip()
            )
    return text.strip()


def _polish_prompt(description: str) -> str:
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": description,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 250},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return _clean_response(r.json()["response"])


@mcp.tool()
def generate_image(
    description: str,
    character: str = "ananya",
    count: int = 1,
    rescue: bool = False,
) -> str:
    """Generate an image of Ananya from a natural language scene description.

    Args:
        description: Plain English scene description, e.g. "her at a rooftop cafe at golden hour"
        character: Character to generate — "ananya" (default) or "kavib"
        count: Number of images to generate (default 1)
        rescue: Use low-VRAM mode (768x1152, 24 steps) if ComfyUI runs out of memory
    """
    if not _ollama_running():
        return "ERROR: Ollama is not running. Start it with: ollama serve"

    try:
        polished = _polish_prompt(description)
    except Exception as e:
        return f"ERROR: Ollama prompt polishing failed: {e}"

    generate_script = Path(__file__).parent / "generate.py"
    cmd = [
        sys.executable, str(generate_script),
        "--prompt", polished,
        "--character", character,
        "--count", str(count),
    ]
    if rescue:
        cmd.append("--rescue")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))

    output_lines = [line for line in result.stdout.splitlines() if "Saved:" in line]
    saved_paths = [line.split("Saved:")[-1].strip() for line in output_lines]

    if result.returncode != 0:
        return f"Polished prompt: {polished}\n\nGeneration failed:\n{result.stderr or result.stdout}"

    paths_str = "\n".join(saved_paths) if saved_paths else "Check output/ folder"
    return f"Polished prompt: {polished}\n\nGenerated {len(saved_paths)} image(s):\n{paths_str}"


@mcp.tool()
def polish_prompt(description: str) -> str:
    """Convert a natural language scene description into a ComfyUI prompt without generating an image.

    Useful for previewing or tweaking prompts before generating.

    Args:
        description: Plain English scene description
    """
    if not _ollama_running():
        return "ERROR: Ollama is not running. Start it with: ollama serve"

    try:
        return _polish_prompt(description)
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
def list_recent_images(character: str = "ananya", limit: int = 5) -> str:
    """List the most recently generated images for a character.

    Args:
        character: Character name — "ananya" or "kavib"
        limit: Number of recent images to list (default 5)
    """
    cfg = _load_config()
    output_dir = ROOT / cfg["paths"]["output_dir"]
    images = sorted(output_dir.rglob(f"{character}_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not images:
        return f"No images found for {character} in {output_dir}"
    recent = images[:limit]
    return "\n".join(str(p) for p in recent)


if __name__ == "__main__":
    mcp.run(transport="stdio")
