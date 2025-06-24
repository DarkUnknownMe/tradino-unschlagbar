# =========================================================================
# TRADINO UNSCHLAGBAR - Production Dockerfile
# Multi-Stage Build für optimierte Production Images
# =========================================================================

# ====================
# STAGE 1: BUILDER
# ====================
FROM python:3.11-slim as builder

LABEL maintainer="TRADINO Development Team"
LABEL version="1.0.0"
LABEL description="Advanced AI Trading System"

# Build Arguments
ARG ENVIRONMENT=production
ARG BUILD_DATE
ARG GIT_COMMIT

# Environment Labels
LABEL build.date=${BUILD_DATE}
LABEL build.environment=${ENVIRONMENT}
LABEL git.commit=${GIT_COMMIT}

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    wget \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create build directory
WORKDIR /build

# Copy and install Python dependencies
COPY requirements.txt requirements-minimal.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# ====================
# STAGE 2: PRODUCTION
# ====================
FROM python:3.11-slim as production

# Create non-root user for security
RUN groupadd -r tradino && useradd -r -g tradino tradino

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /root/.local /home/tradino/.local

# Set PATH for local packages
ENV PATH=/home/tradino/.local/bin:$PATH

# Create application directories
RUN mkdir -p /app/data /app/logs /app/models /app/config \
    && chown -R tradino:tradino /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=tradino:tradino . .

# Create necessary directories and set permissions
RUN mkdir -p \
    /app/data/logs \
    /app/data/backups \
    /app/data/models \
    /app/models \
    /app/logs \
    /app/temp \
    && chown -R tradino:tradino /app \
    && chmod +x /app/main.py

# Switch to non-root user
USER tradino

# Environment variables
ENV PYTHONPATH=/app \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Volume for persistent data
VOLUME ["/app/data", "/app/logs", "/app/models"]

# Default command
CMD ["python", "main.py"]

# ====================
# ALTERNATIVE COMMANDS
# ====================
# Development: CMD ["python", "main.py", "--debug"]
# Dashboard: CMD ["python", "core/trading_dashboard.py"]
# Testing: CMD ["python", "-m", "pytest", "tests/"]
# Monitoring: CMD ["python", "scripts/12h_monitoring_system.py"] 