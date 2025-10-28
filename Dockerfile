# SynFinance Production Dockerfile
# Multi-stage build for optimized image size
# Target: <500MB final image

# Stage 1: Builder - Install dependencies
FROM python:3.12-slim AS builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime - Create minimal production image
FROM python:3.12-slim

# Set metadata
LABEL maintainer="SynFinance Team"
LABEL version="0.6.6"
LABEL description="SynFinance - Synthetic Financial Data Generator with ML & Fraud Detection"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    SYNFINANCE_ENV=production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash synfinance && \
    mkdir -p /app /app/data /app/output /app/logs && \
    chown -R synfinance:synfinance /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=synfinance:synfinance . /app/

# Switch to non-root user
USER synfinance

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: Run FastAPI application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
