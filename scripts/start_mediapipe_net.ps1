<#
.SYNOPSIS
    MediaPipe Net indítása – ZMQ SUB → TCP SENDER (.venv_mediapipe)
.DESCRIPTION
    Ha a .venv_mediapipe nem létezik, először futtasd: .\setup_venvs.ps1 -Target mediapipe
    A tcp_command_server.ps1 és a frame_publisher.ps1 már futnia kell!
.EXAMPLE
    .\start_mediapipe_net.ps1
    .\start_mediapipe_net.ps1 --no-preview
    .\start_mediapipe_net.ps1 --zmq-host 192.168.1.5 --tcp-host 192.168.1.5
#>
$Root   = Split-Path $PSScriptRoot -Parent
$Venv   = Join-Path $Root ".venv_mediapipe"
$Python = Join-Path $Venv "Scripts\python.exe"
$Script = Join-Path $Root "python\vision\mediapipe_net\mediapipe_net.py"

if (-not (Test-Path $Python)) {
    Write-Error ".venv_mediapipe nem található. Futtasd elöbb: .\setup_venvs.ps1 -Target mediapipe"
    exit 1
}

Write-Host "[MEDIAPIPE NET] Indítás: $Script" -ForegroundColor Cyan
Write-Host "[MEDIAPIPE NET] Venv:    $Venv" -ForegroundColor Gray
$env:PYTHONUTF8 = "1"
& $Python $Script @args
