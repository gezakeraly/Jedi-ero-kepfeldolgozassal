<#
.SYNOPSIS
    Frame Publisher indítása – AirSim → ZMQ PUB (.venv_airsim)
.DESCRIPTION
    Ha a .venv_airsim nem létezik, először futtasd: .\setup_venvs.ps1 -Target airsim
    Az UE5 + AirSim szimulációnak futnia kell!
.EXAMPLE
    .\start_publisher.ps1
    .\start_publisher.ps1 --fps 15 --quality 70
    .\start_publisher.ps1 --host 192.168.1.10
#>
$Root   = Split-Path $PSScriptRoot -Parent
$Venv   = Join-Path $Root ".venv_airsim"
$Python = Join-Path $Venv "Scripts\python.exe"
$Script = Join-Path $Root "python\airsim\frame_publisher.py"

if (-not (Test-Path $Python)) {
    Write-Error ".venv_airsim nem található. Futtasd elöbb: .\setup_venvs.ps1 -Target airsim"
    exit 1
}

Write-Host "[PUBLISHER] Indítás: $Script" -ForegroundColor Cyan
Write-Host "[PUBLISHER] Venv:    $Venv" -ForegroundColor Gray
$env:PYTHONUTF8 = "1"
& $Python $Script @args
