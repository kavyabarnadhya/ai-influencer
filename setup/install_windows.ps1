# install_windows.ps1 — One-shot Windows 11 setup for ai-influencer pipeline
# Run from the project root:
#   powershell -ExecutionPolicy Bypass -File setup\install_windows.ps1
#
# Prerequisites: NVIDIA driver installed, Git on PATH, internet access.

$ErrorActionPreference = "Stop"

Write-Host "`n=== AI-Influencer Pipeline Setup ===" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 1. Check CUDA driver
# ---------------------------------------------------------------------------
Write-Host "`n[1/7] Checking CUDA driver..." -ForegroundColor Yellow
try {
    $nvidiaSmi = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    Write-Host "  GPU detected: $nvidiaSmi" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: nvidia-smi not found. Install NVIDIA drivers before continuing." -ForegroundColor Red
    Write-Host "  Download: https://www.nvidia.com/drivers"
    exit 1
}

# ---------------------------------------------------------------------------
# 2. Download + extract ComfyUI Windows Portable
# ---------------------------------------------------------------------------
Write-Host "`n[2/7] Installing ComfyUI Portable..." -ForegroundColor Yellow
$comfyDir = "C:\ComfyUI"

if (Test-Path "$comfyDir\ComfyUI\main.py") {
    Write-Host "  ComfyUI already installed at $comfyDir — skipping download." -ForegroundColor DarkGray
} else {
    $comfyZip = "$env:TEMP\ComfyUI_windows_portable.zip"
    $comfyReleaseUrl = "https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z"

    Write-Host "  Downloading ComfyUI Windows Portable..."
    Write-Host "  NOTE: Download the latest portable release from:"
    Write-Host "        https://github.com/comfyanonymous/ComfyUI/releases/latest"
    Write-Host "  Extract to C:\ComfyUI so that C:\ComfyUI\ComfyUI\main.py exists."
    Write-Host "  Then re-run this script to continue with custom node installation."
    Write-Host ""
    Write-Host "  (Automated download skipped — portable release uses 7z format.)" -ForegroundColor DarkGray
    Write-Host "  After manual extraction, press Enter to continue..." -ForegroundColor Yellow
    Read-Host

    if (-not (Test-Path "$comfyDir\ComfyUI\main.py")) {
        Write-Host "  ERROR: ComfyUI not found at $comfyDir\ComfyUI\main.py" -ForegroundColor Red
        exit 1
    }
}

Write-Host "  ComfyUI found at $comfyDir" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 3. Install custom nodes
# ---------------------------------------------------------------------------
Write-Host "`n[3/7] Installing custom nodes..." -ForegroundColor Yellow
$customNodesDir = "$comfyDir\ComfyUI\custom_nodes"

$customNodes = @(
    @{ Repo = "https://github.com/ltdrdata/ComfyUI-Manager.git";         Dir = "ComfyUI-Manager";         Note = "Node manager" },
    @{ Repo = "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git";     Dir = "ComfyUI-Impact-Pack";     Note = "FaceDetailer core" },
    @{ Repo = "https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git";  Dir = "ComfyUI-Impact-Subpack";  Note = "REQUIRED: UltralyticsDetectorProvider nodes" },
    @{ Repo = "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git";     Dir = "ComfyUI_IPAdapter_plus";  Note = "IP-Adapter SDXL" },
    @{ Repo = "https://github.com/Fannovel16/comfyui_controlnet_aux.git";Dir = "comfyui_controlnet_aux";  Note = "OpenPose preprocessor" },
    @{ Repo = "https://github.com/city96/ComfyUI-GGUF.git";              Dir = "ComfyUI-GGUF";            Note = "GGUF model loader for FLUX" }
)

foreach ($node in $customNodes) {
    $targetDir = "$customNodesDir\$($node.Dir)"
    if (Test-Path $targetDir) {
        Write-Host "  SKIP   $($node.Dir) — already installed" -ForegroundColor DarkGray
    } else {
        Write-Host "  CLONE  $($node.Dir) ($($node.Note))"
        git clone --depth=1 $node.Repo $targetDir
        Write-Host "  OK     $($node.Dir)" -ForegroundColor Green
    }
}

# ---------------------------------------------------------------------------
# 4. Install Impact Pack Python dependencies
# ---------------------------------------------------------------------------
Write-Host "`n[4/7] Installing Impact Pack Python dependencies..." -ForegroundColor Yellow
$comfyPython = "$comfyDir\python_embeds\python.exe"
if (-not (Test-Path $comfyPython)) {
    $comfyPython = "$comfyDir\ComfyUI\python_embeds\python.exe"
}
if (-not (Test-Path $comfyPython)) {
    Write-Host "  WARNING: Cannot find embedded Python. Skipping Impact Pack dep install." -ForegroundColor Yellow
    Write-Host "  Run ComfyUI once; it will auto-install Impact Pack deps on first launch."
} else {
    & $comfyPython -m pip install ultralytics --quiet
    Write-Host "  ultralytics (YOLOv8) installed — no insightface required." -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 5. Create project Python venv + install requirements
# ---------------------------------------------------------------------------
Write-Host "`n[5/7] Setting up project virtual environment..." -ForegroundColor Yellow
$scriptDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$venvDir = "$scriptDir\.venv"

if (Test-Path "$venvDir\Scripts\python.exe") {
    Write-Host "  Venv already exists at $venvDir — skipping creation." -ForegroundColor DarkGray
} else {
    python -m venv $venvDir
    Write-Host "  Created venv at $venvDir" -ForegroundColor Green
}

Write-Host "  Installing requirements.txt..."
& "$venvDir\Scripts\pip.exe" install -r "$scriptDir\requirements.txt" --quiet
Write-Host "  Requirements installed." -ForegroundColor Green

# ---------------------------------------------------------------------------
# 6. Download models
# ---------------------------------------------------------------------------
Write-Host "`n[6/7] Downloading models..." -ForegroundColor Yellow
Write-Host "  This downloads ~20GB total. Runs are resumable if interrupted."
Write-Host "  You will need a HuggingFace token for gated models (FLUX, Juggernaut)."
Write-Host "  Get one at: https://huggingface.co/settings/tokens"
Write-Host ""
$hfToken = Read-Host "  Enter HuggingFace token (or press Enter to skip)"

if ($hfToken) {
    & "$venvDir\Scripts\python.exe" "$scriptDir\setup\download_models.py" `
        --hf-token $hfToken --models-dir "$comfyDir\ComfyUI\models"
} else {
    Write-Host "  Skipping model download. Run later:" -ForegroundColor Yellow
    Write-Host "  .venv\Scripts\python setup\download_models.py --hf-token YOUR_TOKEN"
}

# ---------------------------------------------------------------------------
# 7. Next steps
# ---------------------------------------------------------------------------
Write-Host "`n[7/7] Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Train your character LoRA:"
Write-Host "     Read setup\train_lora_guide.md for step-by-step instructions."
Write-Host "     Then place KaviB_v1_Prod.safetensors in:"
Write-Host "       $comfyDir\ComfyUI\models\loras\"
Write-Host ""
Write-Host "  2. Generate seed images (for LoRA training dataset):"
Write-Host "     .venv\Scripts\python scripts\bootstrap_seeds.py --mode closeup"
Write-Host "     .venv\Scripts\python scripts\bootstrap_seeds.py --mode medium"
Write-Host "     .venv\Scripts\python scripts\bootstrap_seeds.py --mode fullbody"
Write-Host ""
Write-Host "  3. Verify the full setup:"
Write-Host "     .venv\Scripts\python setup\verify_setup.py"
Write-Host ""
Write-Host "  4. Generate images:"
Write-Host "     .venv\Scripts\python scripts\generate.py --prompt 'sitting at a cafe'"
Write-Host ""
Write-Host "See README.md for the full first-run sequence." -ForegroundColor DarkGray
