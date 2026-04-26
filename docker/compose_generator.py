"""
docker-compose.yml
==================
Local testing setup. Runs:
  - purple-agent: this agent (A2A server on port 9010)
  - vllm:         LLM inference server (port 8000) -- requires GPU host

Use `docker-compose --profile cpu up` to run without vLLM (uses HF directly).
"""

# NOTE: This is a YAML file — saved as .py for content, rename to docker-compose.yml
DOCKER_COMPOSE_CONTENT = """
version: "3.9"

services:
  # ── Purple Agent A2A Server ─────────────────────────────────────────────────
  purple-agent:
    build: .
    ports:
      - "9010:9010"
    environment:
      - LLM_BASE_URL=http://vllm:8000
      - MODEL_NAME=deepseek/deepseek-chat-v3-0324
      - MAX_TURNS=15
      - MCTS_BRANCHES=3
      - TEMPERATURE=0.6
      - USE_MCTS=true
    depends_on:
      vllm:
        condition: service_healthy
    networks:
      - agent-net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9010/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ── vLLM Inference Server ───────────────────────────────────────────────────
  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    environment:
      - OPENROUTER_HUB_TOKEN=${OPENROUTER_API_KEY}
    command: >
      --model deepseek/deepseek-chat-v3-0324
      --dtype bfloat16
      --max-model-len 8192
      --gpu-memory-utilization 0.85
      --quantization bitsandbytes
      --trust-remote-code
      --port 8000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - hf_cache:/root/.cache/huggingface
    networks:
      - agent-net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 15s
      timeout: 10s
      retries: 10
      start_period: 120s

networks:
  agent-net:
    driver: bridge

volumes:
  hf_cache:
"""

if __name__ == "__main__":
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker-compose.yml")
    with open(path, "w") as f:
        f.write(DOCKER_COMPOSE_CONTENT.strip())
    print(f"Written to {path}")
