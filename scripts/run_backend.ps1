# Geometra Backend Launcher (PowerShell)
$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$BackendDir = Join-Path $ProjectDir "backend"

Set-Location $BackendDir

# Activate virtual environment
$VenvPath = Join-Path $BackendDir ".venv"
if (-not (Test-Path $VenvPath)) {
    Write-Host "Virtual environment not found. Run scripts\setup.ps1 first." -ForegroundColor Red
    exit 1
}

& "$VenvPath\Scripts\Activate.ps1"

# Create upload/output dirs
New-Item -ItemType Directory -Force -Path "uploads", "outputs" | Out-Null

Write-Host "=== Starting Geometra Backend ===" -ForegroundColor Cyan

# Start Celery worker in background
Write-Host "`n>>> Starting Celery worker..." -ForegroundColor Yellow
$celeryJob = Start-Job -ScriptBlock {
    param($dir, $venv)
    Set-Location $dir
    & "$venv\Scripts\Activate.ps1"
    celery -A geometra.core.task_queue.celery_app worker --loglevel=info --concurrency=1 --pool=solo
} -ArgumentList $BackendDir, $VenvPath

Write-Host "    Celery worker started (Job ID: $($celeryJob.Id))" -ForegroundColor Green

# Start FastAPI server
Write-Host "`n>>> Starting FastAPI server..." -ForegroundColor Yellow
uvicorn geometra.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

# Cleanup
Write-Host "`nStopping Celery worker..." -ForegroundColor Yellow
Stop-Job $celeryJob -ErrorAction SilentlyContinue
Remove-Job $celeryJob -ErrorAction SilentlyContinue
