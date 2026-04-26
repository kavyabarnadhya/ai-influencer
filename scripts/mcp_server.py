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

SYSTEM_PROMPT = """You are a prompt engineer for an AI image generation pipeline. Convert natural language scene descriptions into precise ComfyUI prompt tags for character Ananya.

ISOLATION RULE — strictly enforced:
- NEVER describe physical features: no face, eyes, hair, skin, ethnicity, body shape. The LoRA encodes these.
- ONLY describe: clothing, setting/location, pose, lighting, mood, camera angle, time of day, background elements.
- Output 10–20 comma-separated tags. No sentences. No trigger word. No explanation.

BACKGROUND RULE — always apply:
- Backgrounds must be realistic and partially visible — never fully blurred out.
- Use "f/8.0 sharp realistic background" — the full setting should be clearly visible, almost no blur, like a phone camera shot.
- Always name specific background elements: e.g. "warm wood cafe interior visible behind", "city buildings slightly blurred", "palm trees soft focus".

Examples:
Input: "her at a cafe in the morning looking cozy"
Output: sunlit cafe interior, warm wood tones visible behind, seated by window, oversized knit sweater, iced coffee on table, warm dappled morning light, candid relaxed pose, f/8.0 sharp realistic background, lifestyle portrait

Input: "dramatic rooftop shot at night with city lights"
Output: rooftop terrace, city skyline softly blurred behind, fitted black dress, cinematic side lighting, confident standing pose, moody editorial, cool blue-purple tones, f/8.0 sharp realistic background

Input: "traditional Indian look for a festive occasion"
Output: festive courtyard, string lights and arches visible in background, embroidered silk lehenga, warm golden evening glow, graceful standing pose, elegant Indian fashion editorial, soft ambient light, f/8.0 sharp realistic background"""

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


def _polish_prompt(description: str) -> str:
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": description,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 200},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"].strip()


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
