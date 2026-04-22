#!/usr/bin/env bash
# =============================================================================
# start_vllm.sh — Launch vLLM server on Kaggle 2xT4 GPUs
# =============================================================================
# Run this cell in your Kaggle notebook FIRST, then start the purple agent.
#
# Execution context: Kaggle GPU (2x T4, 16GB each, CUDA 12.x)
# Memory strategy:  4-bit quantisation via bitsandbytes or awq
#
# Usage:
#   bash scripts/start_vllm.sh [--port 8000] [--model MODEL_ID]
# =============================================================================

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.88}"
DTYPE="${DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-bitsandbytes}"   # or "awq" if pre-quantised

# Parse CLI overrides
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)       PORT="$2"; shift 2 ;;
    --model)      MODEL="$2"; shift 2 ;;
    --max_len)    MAX_MODEL_LEN="$2"; shift 2 ;;
    *)            echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "======================================================"
echo " Starting vLLM server"
echo "  model        : $MODEL"
echo "  port         : $PORT"
echo "  max_model_len: $MAX_MODEL_LEN"
echo "  quantization : $QUANTIZATION"
echo "======================================================"

# Install vLLM if not present
if ! python -c "import vllm" &>/dev/null; then
  echo "Installing vLLM…"
  pip install -q vllm bitsandbytes accelerate
fi

# Number of GPUs
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Available GPUs: $NUM_GPUS"

# Tensor parallelism: use all available GPUs
TP="${NUM_GPUS}"

# Launch vLLM in background
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --tensor-parallel-size "$TP" \
  --quantization "$QUANTIZATION" \
  --trust-remote-code \
  --disable-log-requests \
  &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for the server to be ready
echo "Waiting for vLLM to start…"
MAX_WAIT=120
WAITED=0
until curl -sf "http://localhost:${PORT}/health" > /dev/null; do
  sleep 3
  WAITED=$((WAITED + 3))
  if [[ $WAITED -ge $MAX_WAIT ]]; then
    echo "ERROR: vLLM did not start within ${MAX_WAIT}s"
    kill "$VLLM_PID" 2>/dev/null || true
    exit 1
  fi
  echo "  …still waiting (${WAITED}s)"
done

echo "✅ vLLM server ready at http://localhost:${PORT}"
echo "   Model: $MODEL"

# Print PID for later management
echo "$VLLM_PID" > /tmp/vllm.pid
echo "PID saved to /tmp/vllm.pid"
