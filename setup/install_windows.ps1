# AI Influencer Pipeline — Windows 11 Setup Script
# Run from the project root:
#   powershell -ExecutionPolicy Bypass -File setup\install_windows.ps1

param(
    [string]$ComfyUIPath = "C:\ComfyUI"
)

$ErrorActionPreference = "Stop"

Write-Host "`n=== AI Influencer Pipeline Setup ===" -ForegroundColor Cyan

# --- NVIDIA driver check ---
Write-Host "`n[1/5] Checking NVIDIA drivers..." -ForegroundColor Yellow
try {
    $nvidiaSmi = nvidia-smi 2>&1
    Write-Host "  OK: NVIDIA driver detected" -ForegroundColor Green
} catch {
    Write-Host "  WARNING: nvidia-smi not found. Ensure NVIDIA drivers are installed before generating." -ForegroundColor Yellow
}

# --- ComfyUI Portable ---
Write-Host "`n[2/5] Checking ComfyUI Portable at $ComfyUIPath..." -ForegroundColor Yellow
if (Test-Path "$ComfyUIPath\ComfyUI\main.py") {
    Write-Host "  OK: ComfyUI already installed" -ForegroundColor Green
} else {
    Write-Host "  Downloading ComfyUI Portable (this may take several minutes)..." -ForegroundColor Cyan
    $comfyZip = "$env:TEMP\ComfyUI_windows_portable.zip"
    $comfyUrl = "https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z"
    Write-Host "  NOTE: Download the latest ComfyUI Portable from:" -ForegroundColor Yellow
    Write-Host "        https://github.com/comfyanonymous/ComfyUI/releases/latest" -ForegroundColor Cyan
    Write-Host "  Extract to: $ComfyUIPath" -ForegroundColor Yellow
    Write-Host "  Then re-run this script." -ForegroundColor Yellow
    Write-Host "`n  Alternatively, install via the ComfyUI Desktop app:" -ForegroundColor Yellow
    Write-Host "  https://www.comfy.org/download" -ForegroundColor Cyan
    exit 1
}

# --- Custom nodes ---
Write-Host "`n[3/5] Installing custom nodes..." -ForegroundColor Yellow
$customNodesPath = "$ComfyUIPath\ComfyUI\custom_nodes"

$nodes = @(
    @{ Name = "ComfyUI-Manager";       Repo = "https://github.com/ltdrdata/ComfyUI-Manager.git" },
    @{ Name = "ComfyUI-Impact-Pack";   Repo = "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git" },
    @{ Name = "ComfyUI-Impact-Subpack";Repo = "https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git" },
    @{ Name = "ComfyUI_IPAdapter_plus";Repo = "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git" },
    @{ Name = "ComfyUI-Advanced-ControlNet"; Repo = "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git" },
    @{ Name = "ComfyUI-GGUF";          Repo = "https://github.com/city96/ComfyUI-GGUF.git" }
)

foreach ($node in $nodes) {
    $nodePath = Join-Path $customNodesPath $node.Name
    if (Test-Path $nodePath) {
        Write-Host "  OK: $($node.Name) already installed" -ForegroundColor Green
    } else {
        Write-Host "  Cloning $($node.Name)..." -ForegroundColor Cyan
        git clone $node.Repo $nodePath
        Write-Host "  OK: $($node.Name)" -ForegroundColor Green
    }
}

# --- Python venv ---
Write-Host "`n[4/5] Creating Python virtual environment..." -ForegroundColor Yellow
$projectRoot = Split-Path $PSScriptRoot -Parent
$venvPath = Join-Path $projectRoot ".venv"

if (Test-Path "$venvPath\Scripts\python.exe") {
    Write-Host "  OK: .venv already exists" -ForegroundColor Green
} else {
    python -m venv $venvPath
    Write-Host "  OK: .venv created" -ForegroundColor Green
}

# --- Install Python dependencies ---
Write-Host "`n[5/5] Installing Python dependencies..." -ForegroundColor Yellow
$pip = Join-Path $venvPath "Scripts\pip.exe"
$requirements = Join-Path $projectRoot "requirements.txt"
& $pip install -r $requirements --quiet
Write-Host "  OK: dependencies installed" -ForegroundColor Green

# --- Done ---
Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Get a Hugging Face token: https://huggingface.co/settings/tokens" -ForegroundColor White
Write-Host "  2. Activate venv:   .venv\Scripts\activate" -ForegroundColor White
Write-Host "  3. Download models: python setup\download_models.py --hf-token hf_YOUR_TOKEN" -ForegroundColor White
Write-Host "  4. Start ComfyUI:   $ComfyUIPath\run_nvidia_gpu.bat" -ForegroundColor White
Write-Host "  5. Verify setup:    python setup\verify_setup.py" -ForegroundColor White
Write-Host ""
