# =============================================================================
# Purple Agent – Dockerfile
# =============================================================================
# Two-stage build:
#   Stage 1: builder   – install heavy ML deps
#   Stage 2: runtime   – lean image with just what's needed to serve
#
# The model is NOT bundled here — it is pulled from HuggingFace at runtime
# OR you point LLM_BASE_URL at a separately-running vLLM instance.
#
# For AgentBeats competition, the recommended setup is:
#   - Purple agent container: this image (serves A2A on port 9022)
#   - LLM server: separate vLLM container (or cloud GPU endpoint)
#
# Build:
#   docker build -t purple-agent:latest .
#
# Run (with external vLLM):
#   docker run -p 9022:9022 \
#     -e LLM_BASE_URL=http://your-vllm-host:8000 \
#     -e MODEL_NAME=google/gemini-3-flash-preview \
#     purple-agent:latest
#
# Run (with local HuggingFace, requires GPU):
#   docker run --gpus all -p 9022:9022 \
#     -e LLM_BASE_URL=local \
#     -e MODEL_NAME=google/gemini-3-flash-preview \
#     purple-agent:latest
# =============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=9022

WORKDIR /app

# Install system dependencies + official Docker CLI
# Docker CLI is required for Docker-out-of-Docker (sibling container spawning)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc python3-dev ca-certificates gnupg && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg \
         -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    . /etc/os-release && \
    echo "deb [arch=$(dpkg --print-architecture) \
         signed-by=/etc/apt/keyrings/docker.asc] \
         https://download.docker.com/linux/debian $VERSION_CODENAME stable" \
    > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

# Copy build config and source
COPY pyproject.toml .
COPY src/ ./src/
COPY tests/ ./tests/

# Copy scripts directory — || true guards against empty scripts/ dir
COPY scripts/ ./scripts/
# Fix: glob fails if no .sh files exist — || true makes it non-fatal
RUN chmod +x scripts/*.sh 2>/dev/null || true

# Install all dependencies from pyproject.toml
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 9022

# Bind to 0.0.0.0 — mandatory for container networking
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "9022"]