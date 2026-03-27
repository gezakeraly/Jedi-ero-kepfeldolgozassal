<#
.SYNOPSIS
    Webcam Gesture Net indítása – laptop kamera → TCP SENDER (.venv_mediapipe)
.DESCRIPTION
    A tcp_command_server.ps1 már futnia kell!
    A frame_publisher.ps1 NEM szükséges – közvetlenül a laptop kameráját használja.
.EXAMPLE
    .\start_webcam_net.ps1
    .\start_webcam_net.ps1 --no-preview
    .\start_webcam_net.ps1 --camera 1        # ha nem a 0-s kamera kell
    .\start_webcam_net.ps1 --tcp-host 192.168.1.5
#>
$Root   = Split-Path $PSScriptRoot -Parent
$Venv   = Join-Path $Root ".venv_mediapipe"
$Python = Join-Path $Venv "Scripts\python.exe"
$Script = Join-Path $Root "python\vision\mediapipe_net\webcam_net.py"

if (-not (Test-Path $Python)) {
    Write-Error ".venv_mediapipe nem található. Futtasd elöbb: .\setup_venvs.ps1 -Target mediapipe"
    exit 1
}

Write-Host "[WEBCAM NET] Indítás: $Script" -ForegroundColor Cyan
Write-Host "[WEBCAM NET] Venv:    $Venv" -ForegroundColor Gray
$env:PYTHONUTF8 = "1"
& $Python $Script @args
