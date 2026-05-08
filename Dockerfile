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
#   - Purple agent container: this image (serves A2A on port 9010)
#   - LLM server: separate vLLM container (or cloud GPU endpoint)
#
# Build:
#   docker build -t purple-agent:latest .
#
# Run (with external vLLM):
#   docker run -p 9010:9010 \
#     -e LLM_BASE_URL=http://your-vllm-host:8000 \
#     -e MODEL_NAME=deepseek/deepseek-v4-flash \
#     purple-agent:latest
#
# Run (with local HuggingFace, requires GPU):
#   docker run --gpus all -p 9010:9010 \
#     -e LLM_BASE_URL=local \
#     -e MODEL_NAME=deepseek/deepseek-v4-flash \
#     purple-agent:latest
# =============================================================================

# Use your preferred 3.11-slim base
FROM python:3.11-slim

# Prevent Python from writing pyc files and keep stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=9010

WORKDIR /app

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    python3-dev \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy the manifest and project files
COPY pyproject.toml .

# 3. COPY THE CODE FIRST (Mandatory for Hatchling build metadata)
COPY src/ ./src/
COPY scripts/ ./scripts/

# 4. Install Python Dependencies and the local project
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# 5. Ensure utility scripts are executable
RUN chmod +x scripts/*.sh

# 6. Boot the server
EXPOSE 9010
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "9010"]