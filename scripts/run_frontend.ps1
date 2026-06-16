# Geometra Frontend Launcher (PowerShell)
$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$FrontendDir = Join-Path $ProjectDir "frontend"

Set-Location $FrontendDir

if (-not (Test-Path "node_modules")) {
    Write-Host "node_modules not found. Installing dependencies..." -ForegroundColor Yellow
    npm install
}

Write-Host "=== Starting Geometra Frontend ===" -ForegroundColor Cyan
npx vite --host 0.0.0.0 --port 5173
