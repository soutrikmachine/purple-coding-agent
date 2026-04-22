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
#     -e MODEL_NAME=Qwen/Qwen2.5-Coder-7B-Instruct \
#     purple-agent:latest
#
# Run (with local HuggingFace, requires GPU):
#   docker run --gpus all -p 9010:9010 \
#     -e LLM_BASE_URL=local \
#     -e MODEL_NAME=Qwen/Qwen2.5-Coder-7B-Instruct \
#     purple-agent:latest
# =============================================================================

FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .

# 1. Install PyTorch specifically from the CPU index
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install all other standard dependencies from the default PyPI index
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    requests \
    huggingface-hub

# Transformers (CPU fallback — GPU handled by separate vLLM service)
RUN pip install --no-cache-dir transformers accelerate

# ── Runtime Stage ─────────────────────────────────────────────────────────────

# 1. Use Python 3.11 Slim
FROM python:3.11-slim

# 2. Install curl (Required for your HEALTHCHECK line to work!)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Add the requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your source code
# We copy the 'src' folder specifically so the paths match your CMD
COPY src/ ./src/

# 5. Environment Variables
ENV PYTHONPATH=/app/src
ENV PORT=9010
ENV LLM_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-Coder-7B-Instruct
ENV PYTHONUNBUFFERED=1

EXPOSE 9010

# 6. Healthcheck (Now it will work because we installed curl)
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:9010/health || exit 1

# 7. Start the server
CMD ["python", "src/server.py"]
