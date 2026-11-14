# ==============================================================================
# Production Dockerfile for Healthcare RAG & Document Ingestion API
# Multi-stage optimized, non-root user, healthcheck-ready
# ==============================================================================

# Stage 1: Build React Frontend Dashboard
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Build Environment
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies into wheels/cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


# ==============================================================================
# Final Runtime Stage
# ==============================================================================
FROM python:3.12-slim AS runner

WORKDIR /app

# Install runtime system packages (libpq for postgres, curl for healthcheck, poppler for fitz/pdf)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create application user and group for security
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -m -s /bin/bash appuser

# Copy and install python wheels
COPY --from=builder /app/wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Download spaCy medical/general NLP model for PII redaction
RUN python -m spacy download en_core_web_sm

# Copy application source code and configuration
COPY --chown=appuser:appgroup app /app/app
COPY --chown=appuser:appgroup alembic /app/alembic
COPY --chown=appuser:appgroup alembic.ini /app/alembic.ini
COPY --chown=appuser:appgroup scripts /app/scripts
COPY --chown=appuser:appgroup litellm_config.yaml /app/litellm_config.yaml
COPY --from=frontend-builder --chown=appuser:appgroup /frontend/dist /app/frontend/dist

RUN chmod +x /app/scripts/*.sh

# Ensure writable data and log directories
RUN mkdir -p /app/data/uploads /app/logs && \
    chown -R appuser:appgroup /app/data /app/logs

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000

USER appuser

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health/liveness || exit 1

# Production entrypoint and default command
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["api"]

