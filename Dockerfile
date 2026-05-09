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
#     -e MODEL_NAME=deepseek/deepseek-v4-flash \
#     purple-agent:latest
#
# Run (with local HuggingFace, requires GPU):
#   docker run --gpus all -p 9022:9022 \
#     -e LLM_BASE_URL=local \
#     -e MODEL_NAME=deepseek/deepseek-v4-flash \
#     purple-agent:latest
# =============================================================================

FROM python:3.11-slim

# Fix: Set PYTHONPATH to /app so 'src.server:app' is findable
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=9022

WORKDIR /app

# The Elite Fix: Official Docker CLI installation
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc python3-dev ca-certificates gnupg && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    # Dynamically grab the Debian version codename (e.g., bookworm/bullseye) for the repo
    . /etc/os-release && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $VERSION_CODENAME stable" \
    > /etc/apt/sources.list.d/docker.list && \
    apt-get update && apt-get install -y --no-install-recommends docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

# Copy build config and code
COPY pyproject.toml .
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install with uvicorn[standard]
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

RUN chmod +x scripts/*.sh

EXPOSE 9022

# Ensure we bind to 0.0.0.0 (mandatory for container networking)
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "9022"]