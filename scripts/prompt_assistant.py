import subprocess
import sys
from pathlib import Path

import click
import requests
from rich.console import Console
from rich.panel import Panel

console = Console()
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
- NEVER drop clothing from the prompt even if the scene description doesn't mention it. Invent something fitting the setting.

LIGHTING RULE — always be specific:
- Bad: "morning light", "golden light", "good lighting"
- Good: "soft diffused north window light", "warm golden hour backlight", "overcast flat outdoor light", "dramatic single side split lighting", "warm amber tungsten bedside lamp", "cool blue dusk ambient light", "harsh noon sun overhead"

BACKGROUND RULE — always apply:
- Name specific visible elements: "warm timber cafe walls with pendant lights", "floor-to-ceiling glass overlooking Gurgaon skyline", "neon-lit street signs reflecting on wet pavement", "terracotta courtyard with terracotta pots"
- Add "f/8.0 sharp realistic background" — fully visible, like a phone camera, no studio blur.

FRAMING RULE — match shot type exactly, always include one:
- Full body: "full body shot, head to toe, legs and feet visible"
- Waist-up: "waist-up shot, medium framing"
- Close-up: "close-up portrait, face and shoulders only"
- Default to waist-up if unspecified.

HAND RULE — for full body and waist-up only (skip for close-up):
- Prefer hidden: "hands in coat pockets", "arms relaxed at sides", "holding bag strap", "one hand on railing"
- For seated: "hands resting on table edge, partially out of frame"
- NEVER: hands near face, fingers touching face.

EDITORIAL QUALITY — always end with one style tag:
- "candid Instagram editorial", "high fashion editorial", "lifestyle portrait", "street style editorial", "travel editorial", "intimate editorial portrait"

Examples:

User: "close-up at a sunlit Delhi cafe, white linen top, morning"
Output: close-up portrait, face and shoulders only, sunlit cafe interior, warm timber walls with pendant lights visible behind, fitted white linen V-neck top, soft diffused north window light, warm golden morning tones, direct relaxed gaze, f/8.0 sharp realistic background, lifestyle portrait

User: "full body rooftop golden hour, silk slip dress, city skyline"
Output: full body shot, head to toe, legs and feet visible, rooftop terrace, Gurgaon glass towers softly visible behind, champagne silk bias-cut slip dress, warm golden hour backlight, wind in hair, arms relaxed at sides, long shadow on rooftop floor, f/8.0 sharp realistic background, high fashion editorial

User: "night city street, black bodycon dress, neon lights"
Output: full body shot, head to toe, legs and feet visible, rain-wet city pavement, neon signs reflecting on ground, fitted black ribbed bodycon mini dress, high heels, hands in jacket pockets, cool blue-purple neon ambient light, moody dramatic, f/8.0 sharp realistic background, street style editorial

User: "hotel room evening, champagne satin slip, sitting on bed"
Output: waist-up shot, medium framing, luxury hotel suite, warm amber tungsten bedside lamp, ivory bedding and curtains visible, champagne satin thin-strap slip dress, seated on bed edge, arms relaxed at sides, intimate warm tones, soft shadow on far wall, f/8.0 sharp realistic background, intimate editorial portrait

User: "Goa beach afternoon, white linen outfit, travel vibe"
Output: waist-up shot, medium framing, Goa beach promenade, turquoise sea and palm trees visible behind, white linen wide-leg trousers, white linen shirt knotted at waist, overcast flat beach light, loose hair in sea breeze, one hand on hip, f/8.0 sharp realistic background, travel editorial"""

HAND_POSITION_TAGS = (
    "hand",
    "hands",
    "arm",
    "arms",
    "pocket",
    "pockets",
    "strap",
    "crossed",
)


def ensure_hand_position(prompt: str) -> str:
    """Add a hand-position tag if the local LLM forgets the rule."""
    prompt_lower = prompt.lower()
    if any(tag in prompt_lower for tag in HAND_POSITION_TAGS):
        return prompt

    seated_context = ("seated", "sitting", "cafe", "coffee", "table", "restaurant")
    hand_tag = (
        "both hands out of frame below table"
        if any(tag in prompt_lower for tag in seated_context)
        else "hands not visible"
    )
    return f"{prompt}, {hand_tag}"


def ollama_running() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def model_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        return any(MODEL in m for m in models)
    except Exception:
        return False


def _clean_response(text: str) -> str:
    """Strip any preamble lines the model adds before the actual tags."""
    lines = text.strip().splitlines()
    for i, line in enumerate(lines):
        # First line that looks like comma-separated tags (contains a comma, no colon)
        if "," in line and ":" not in line:
            return ", ".join(
                part.strip()
                for part in " ".join(lines[i:]).split(",")
                if part.strip()
            )
    return text.strip()


def polish_prompt(user_input: str) -> str:
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": user_input,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 250},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    return _clean_response(r.json()["response"])


@click.command()
@click.argument("description", required=False)
@click.option("--character", default="ananya", show_default=True, help="Character [ananya|kavib]")
@click.option("--count", default=1, show_default=True, help="Number of images to generate")
@click.option("--rescue", is_flag=True, help="Low-VRAM mode")
@click.option("--dry-run", is_flag=True, help="Print polished prompt only, do not generate")
@click.option("--review", is_flag=True, help="Review loop: edit → re-polish → approve before generating")
@click.option("--seed", default=None, type=int, help="Seed for generation")
def main(description: str | None, character: str, count: int, rescue: bool, dry_run: bool, review: bool, seed: int | None):
    """Convert natural language to an Ananya prompt and generate an image.

    DESCRIPTION: Natural language scene description, e.g. "her at a cafe in the morning"
    If omitted, enters interactive mode.
    """
    if not ollama_running():
        console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
        raise SystemExit(1)

    if not model_available():
        console.print(f"[yellow]Model {MODEL} not found. Pulling now...[/yellow]")
        subprocess.run(["ollama", "pull", MODEL], check=True)

    if description is None:
        console.print("[cyan]Interactive mode — type your scene description (Ctrl+C to exit)[/cyan]")
        while True:
            try:
                description = console.input("\n[bold]Scene:[/bold] ").strip()
                if not description:
                    continue
                _run_once(description, character, count, rescue, dry_run, review, seed)
            except KeyboardInterrupt:
                console.print("\n[dim]Exiting.[/dim]")
                break
    else:
        _run_once(description, character, count, rescue, dry_run, review, seed)


def _run_once(description: str, character: str, count: int, rescue: bool, dry_run: bool, review: bool, seed: int | None):
    console.print(f"\n[dim]Polishing prompt...[/dim]")
    polished = ensure_hand_position(polish_prompt(description))

    console.print(Panel(polished, title="[green]Polished prompt[/green]", border_style="green"))

    if review:
        console.print("[dim]Edit the prompt below (press Enter twice to keep as-is, or type changes):[/dim]")
        console.print("[dim]You can add tags, remove tags, or describe changes in plain English.[/dim]\n")
        user_edit = console.input("[bold yellow]Your edit:[/bold yellow] ").strip()
        if user_edit:
            console.print("\n[dim]Re-polishing with your edits...[/dim]")
            refine_input = f"Original prompt: {polished}\nUser changes: {user_edit}\nOutput a refined final prompt incorporating the changes."
            polished = ensure_hand_position(polish_prompt(refine_input))
            console.print(Panel(polished, title="[green]Refined prompt[/green]", border_style="green"))

        approve = console.input("\n[bold]Generate with this prompt? [Y/n]:[/bold] ").strip().lower()
        if approve == "n":
            console.print("[dim]Cancelled.[/dim]")
            return

    if dry_run:
        return

    generate_script = Path(__file__).parent / "generate.py"
    cmd = [sys.executable, str(generate_script), "--prompt", polished, "--character", character, "--count", str(count)]
    if rescue:
        cmd.append("--rescue")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
