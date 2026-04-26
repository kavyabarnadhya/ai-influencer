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

SYSTEM_PROMPT = """You are a prompt engineer for an AI image generation pipeline. Your job is to convert natural language scene descriptions into precise ComfyUI prompts for a character named Ananya.

STRICT RULES — the Isolation Rule:
1. NEVER describe physical features: no face shape, eye color, hair, skin tone, ethnicity, body shape. The LoRA handles all of that.
2. ONLY describe: clothing, setting/location, pose, lighting, mood, camera angle, time of day, background.
3. Keep it concise: 10–20 comma-separated tags. No sentences.
4. Do NOT include the trigger word "AnanyaAI" — the pipeline adds it automatically.
5. Output ONLY the prompt tags. No explanation, no preamble, no quotes.

Examples:
User: "her at a cafe in the morning looking cozy"
Output: sunlit cafe interior, seated by window, oversized knit sweater, iced coffee on table, warm dappled morning light, candid relaxed pose, soft bokeh background, lifestyle portrait

User: "dramatic rooftop shot at night with city lights"
Output: rooftop terrace, nighttime, city bokeh background, fitted black dress, cinematic side lighting, confident standing pose, moody editorial, cool blue-purple tones

User: "traditional Indian look for a festive occasion"
Output: festive courtyard, embroidered silk lehenga, warm string lights, golden evening glow, graceful standing pose, elegant Indian fashion editorial, soft ambient light"""


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


def polish_prompt(user_input: str) -> str:
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": user_input,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 200},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["response"].strip()


@click.command()
@click.argument("description", required=False)
@click.option("--character", default="ananya", show_default=True, help="Character [ananya|kavib]")
@click.option("--count", default=1, show_default=True, help="Number of images to generate")
@click.option("--rescue", is_flag=True, help="Low-VRAM mode")
@click.option("--dry-run", is_flag=True, help="Print polished prompt only, do not generate")
@click.option("--seed", default=None, type=int, help="Seed for generation")
def main(description: str | None, character: str, count: int, rescue: bool, dry_run: bool, seed: int | None):
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
                _run_once(description, character, count, rescue, dry_run, seed)
            except KeyboardInterrupt:
                console.print("\n[dim]Exiting.[/dim]")
                break
    else:
        _run_once(description, character, count, rescue, dry_run, seed)


def _run_once(description: str, character: str, count: int, rescue: bool, dry_run: bool, seed: int | None):
    console.print(f"\n[dim]Polishing prompt...[/dim]")
    polished = polish_prompt(description)

    console.print(Panel(polished, title="[green]Polished prompt[/green]", border_style="green"))

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
