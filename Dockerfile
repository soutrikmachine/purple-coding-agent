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

# 1. Install Phase 2 System Dependencies
# - gcc & python3-dev: Mandatory for tree-sitter (Graph RAG)
# - docker.io: Mandatory for sibling container execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    python3-dev \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies[cite: 1]
# We've moved from requirements.txt to the more robust pyproject.toml
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# 3. Copy Application Code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Ensure utility scripts are executable
RUN chmod +x scripts/*.sh

# 4. A2A Handshake Port[cite: 2]
EXPOSE 9010

# 5. Boot the server using the A2A handshake logic[cite: 2]
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "9010"]
