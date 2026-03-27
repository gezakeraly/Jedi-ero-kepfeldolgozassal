<#
.SYNOPSIS
    Létrehozza a virtuális Python környezeteket és telepíti a csomagokat.

.DESCRIPTION
    Két venv jön létre:
        .venv_airsim     - tcp_command_server, car_controller, frame_publisher
        .venv_mediapipe  - python/vision/mediapipe_net/ (rögzített, érzékeny verziók!)

    Mindkettő a rendszer Python 3.8.8-at használja.

.PARAMETER Target
    Melyik venvet hozza létre: "airsim", "mediapipe", vagy "all" (alapértelmezett)

.EXAMPLE
    .\setup_venvs.ps1
    .\setup_venvs.ps1 -Target mediapipe
    .\setup_venvs.ps1 -Target airsim
#>

param(
    [ValidateSet("airsim", "mediapipe", "all")]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"
$Root     = Split-Path $PSScriptRoot -Parent
$Python38 = "C:\Users\szebe\AppData\Local\Programs\Python\Python38\python.exe"

# ── Ellenőrzés ──────────────────────────────────────────────────────────────
if (-not (Test-Path $Python38)) {
    Write-Error "Python 3.8 nem található: $Python38`nEllenőrizd az elérési utat!"
    exit 1
}

Write-Host "Python 3.8: $Python38" -ForegroundColor Cyan
& $Python38 --version


function Create-Venv {
    param([string]$Name, [string]$ReqFile)

    $VenvPath = Join-Path $Root $Name
    Write-Host "`n═══════════════════════════════════════" -ForegroundColor DarkCyan
    Write-Host "  Venv: $Name" -ForegroundColor Yellow
    Write-Host "═══════════════════════════════════════" -ForegroundColor DarkCyan

    if (Test-Path $VenvPath) {
        Write-Host "  Már létezik, kihagyom a létrehozást." -ForegroundColor Gray
    } else {
        Write-Host "  Létrehozás (--system-site-packages)..." -ForegroundColor Green
        # --system-site-packages: örökli a rendszer Python csomagjait (tensorflow, mediapipe, cosysairsim stb.)
        # Csak a hiányzó extra csomagokat (pl. pyzmq) kell telepíteni
        & $Python38 -m venv --system-site-packages $VenvPath
        Write-Host "  OK: $VenvPath" -ForegroundColor Green
    }

    $pip = Join-Path $VenvPath "Scripts\pip.exe"
    Write-Host "  Pip frissítése..." -ForegroundColor Gray
    & $pip install --upgrade pip --quiet

    Write-Host "  Csomagok telepítése: $ReqFile" -ForegroundColor Green
    & $pip install -r $ReqFile
    Write-Host "  KÉSZ: $Name`n" -ForegroundColor Green
}


# ── AirSim venv ────────────────────────────────────────────────────────────
if ($Target -eq "airsim" -or $Target -eq "all") {
    $req = Join-Path $Root "python\requirements_airsim.txt"
    Create-Venv -Name ".venv_airsim" -ReqFile $req
}

# ── MediaPipe venv ─────────────────────────────────────────────────────────
if ($Target -eq "mediapipe" -or $Target -eq "all") {
    $req = Join-Path $Root "python\vision\mediapipe_net\requirements.txt"
    Create-Venv -Name ".venv_mediapipe" -ReqFile $req
}

Write-Host "Minden venv elkészült." -ForegroundColor Cyan
Write-Host "Indítási scriptek: scripts\start_server.ps1 | scripts\start_publisher.ps1 | scripts\start_mediapipe_net.ps1 | scripts\start_car.ps1"
