#!/usr/bin/env bash
# ==============================================================================
# Production Container Entrypoint Script
# Waits for backing services, runs database migrations, and boots application
# ==============================================================================

set -euo pipefail

echo "==> [Entrypoint] Starting Healthcare RAG Service Container..."

# Helper: Wait for a TCP socket to become available using python
wait_for_service() {
    local host="$1"
    local port="$2"
    local name="$3"
    local max_retries="${4:-30}"
    local count=0

    echo "==> [Entrypoint] Waiting for ${name} at ${host}:${port}..."
    until python -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1.0)
try:
    s.connect(('${host}', int(${port})))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; do
        count=$((count + 1))
        if [ "$count" -ge "$max_retries" ]; then
            echo "==> [Entrypoint] ERROR: Timed out waiting for ${name} at ${host}:${port}" >&2
            exit 1
        fi
        sleep 1
    done
    echo "==> [Entrypoint] ${name} is UP and ready!"
}

# 1. Parse and wait for PostgreSQL if configured
if [ -n "${DATABASE_URL:-}" ] && [[ "${DATABASE_URL}" == *"postgres"* ]]; then
    DB_TARGET=$(python -c "
import urllib.parse, os
parsed = urllib.parse.urlsplit(os.environ['DATABASE_URL'])
if parsed.hostname:
    print(f'{parsed.hostname} {parsed.port or 5432}')
" 2>/dev/null || true)
    if [ -n "${DB_TARGET}" ]; then
        DB_HOST=$(echo "$DB_TARGET" | awk '{print $1}')
        DB_PORT=$(echo "$DB_TARGET" | awk '{print $2}')
        wait_for_service "${DB_HOST}" "${DB_PORT}" "PostgreSQL Database" 30
    fi
fi

# 2. Parse and wait for Redis if configured
if [ -n "${REDIS_URL:-}" ] && [[ "${REDIS_URL}" == *"redis"* ]]; then
    REDIS_TARGET=$(python -c "
import urllib.parse, os
parsed = urllib.parse.urlsplit(os.environ['REDIS_URL'])
if parsed.hostname:
    print(f'{parsed.hostname} {parsed.port or 6379}')
" 2>/dev/null || true)
    if [ -n "${REDIS_TARGET}" ]; then
        REDIS_HOST=$(echo "$REDIS_TARGET" | awk '{print $1}')
        REDIS_PORT=$(echo "$REDIS_TARGET" | awk '{print $2}')
        wait_for_service "${REDIS_HOST}" "${REDIS_PORT}" "Redis Service" 30
    fi
fi

# 3. Apply Alembic Database Migrations if running the API web server
if [[ "${1:-api}" == "api" || "${1:-}" == "uvicorn" ]]; then
    echo "==> [Entrypoint] Running database migrations via Alembic..."
    alembic upgrade head
    echo "==> [Entrypoint] Database schema is up to date!"
fi

# 4. Dispatch service based on arguments
LOG_LEVEL_LOWER=$(echo "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')

if [[ "${1:-api}" == "api" ]]; then
    echo "==> [Entrypoint] Booting FastAPI API Server with Uvicorn..."
    WORKERS="${WEB_CONCURRENCY:-1}"
    if [ "${WORKERS}" -gt 1 ]; then
        exec uvicorn app.main:app \
            --host 0.0.0.0 \
            --port "${PORT:-8000}" \
            --workers "${WORKERS}" \
            --log-level "${LOG_LEVEL_LOWER}"
    else
        exec uvicorn app.main:app \
            --host 0.0.0.0 \
            --port "${PORT:-8000}" \
            --log-level "${LOG_LEVEL_LOWER}"
    fi
elif [[ "${1:-}" == "worker" ]]; then
    echo "==> [Entrypoint] Booting Celery Ingestion Worker..."
    exec celery -A app.ingestion.celery_app.celery_app worker \
        --loglevel="${LOG_LEVEL_LOWER}" \
        --concurrency="${CELERY_CONCURRENCY:-2}" \
        -Q celery,document_ingestion
else
    # Execute any custom command passed to docker run
    exec "$@"
fi

