<#
.SYNOPSIS
    Car Controller indítása (.venv_airsim)
.DESCRIPTION
    Ha a .venv_airsim nem létezik, először futtasd: .\setup_venvs.ps1 -Target airsim
    Az UE5 + AirSim szimulációnak futnia kell!
.EXAMPLE
    .\start_car.ps1
    .\start_car.ps1 --vehicle Car1
#>
$Root   = Split-Path $PSScriptRoot -Parent
$Venv   = Join-Path $Root ".venv_airsim"
$Python = Join-Path $Venv "Scripts\python.exe"
$Script = Join-Path $Root "python\control\car_controller.py"

if (-not (Test-Path $Python)) {
    Write-Error ".venv_airsim nem található. Futtasd elöbb: .\setup_venvs.ps1 -Target airsim"
    exit 1
}

Write-Host "[CAR] Indítás: $Script" -ForegroundColor Cyan
Write-Host "[CAR] Venv:    $Venv" -ForegroundColor Gray
$env:PYTHONUTF8 = "1"
& $Python $Script @args
