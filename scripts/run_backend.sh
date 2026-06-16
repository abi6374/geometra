#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_DIR/backend"

cd "$BACKEND_DIR"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found. Run scripts/setup.sh first."
    exit 1
fi

# Create upload/output dirs
mkdir -p uploads outputs

echo "=== Starting Geometra Backend ==="
echo ""

# Start Celery worker in background
echo ">>> Starting Celery worker..."
celery -A geometra.core.task_queue.celery_app worker \
    --loglevel=info \
    --concurrency=1 \
    --pool=solo &
CELERY_PID=$!
echo "    Celery worker PID: $CELERY_PID"

# Start FastAPI server
echo ">>> Starting FastAPI server..."
uvicorn geometra.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info

# Cleanup Celery on exit
kill $CELERY_PID 2>/dev/null || true
