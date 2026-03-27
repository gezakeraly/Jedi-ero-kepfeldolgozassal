<#
.SYNOPSIS
    TCP Command Server indítása (.venv_airsim)
.DESCRIPTION
    Ha a .venv_airsim nem létezik, először futtasd: .\setup_venvs.ps1 -Target airsim
#>
$Root   = Split-Path $PSScriptRoot -Parent
$Venv   = Join-Path $Root ".venv_airsim"
$Python = Join-Path $Venv "Scripts\python.exe"
$Script = Join-Path $Root "python\control\tcp_server.py"

if (-not (Test-Path $Python)) {
    Write-Error ".venv_airsim nem található. Futtasd elöbb: .\setup_venvs.ps1 -Target airsim"
    exit 1
}

Write-Host "[SERVER] Indítás: $Script" -ForegroundColor Cyan
Write-Host "[SERVER] Venv:    $Venv" -ForegroundColor Gray
$env:PYTHONUTF8 = "1"
& $Python $Script @args
