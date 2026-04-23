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

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONPATH=/app/src
ENV PORT=9010
ENV LLM_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-Coder-7B-Instruct
ENV PYTHONUNBUFFERED=1

EXPOSE 9010

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:9010/health || exit 1

CMD ["python", "src/server.py"]
