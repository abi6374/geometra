#!/usr/bin/env bash
set -euo pipefail

echo "=== Geometra Local Development Setup ==="

# ── Backend ──
echo ""
echo ">>> Setting up Python backend..."
cd backend

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "    Created virtual environment"
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
echo "    Backend dependencies installed"

# Copy env file if not present
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "    Created .env from .env.example"
fi

cd ..

# ── Frontend ──
echo ""
echo ">>> Setting up frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
    npm install
    echo "    Frontend dependencies installed"
else
    echo "    node_modules already exists, skipping install"
fi

cd ..

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the application, run:"
echo "  scripts/run_backend.sh   (starts FastAPI + Celery worker)"
echo "  scripts/run_frontend.sh  (starts Vite dev server)"
