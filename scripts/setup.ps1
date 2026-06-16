# Geometra Local Development Setup (PowerShell)
Write-Host "=== Geometra Local Development Setup ===" -ForegroundColor Cyan

# Backend
Write-Host "`n>>> Setting up Python backend..." -ForegroundColor Yellow
Set-Location backend

if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "    Created virtual environment" -ForegroundColor Green
}

& ".venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -e ".[dev]"
Write-Host "    Backend dependencies installed" -ForegroundColor Green

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "    Created .env from .env.example" -ForegroundColor Green
}

Set-Location ..

# Frontend
Write-Host "`n>>> Setting up frontend..." -ForegroundColor Yellow
Set-Location frontend

if (-not (Test-Path "node_modules")) {
    npm install
    Write-Host "    Frontend dependencies installed" -ForegroundColor Green
} else {
    Write-Host "    node_modules already exists, skipping install" -ForegroundColor Yellow
}

Set-Location ..

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "`nTo start the application, run in separate terminals:"
Write-Host "  scripts\run_backend.ps1   (starts FastAPI + Celery worker)"
Write-Host "  scripts\run_frontend.ps1  (starts Vite dev server)"
