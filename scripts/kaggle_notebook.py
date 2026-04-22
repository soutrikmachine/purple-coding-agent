# Purple Agent – Kaggle 2xT4 Notebook
# =====================================
# Run each cell in order. GPU: 2x T4 (recommended)
#
# WEEK 1 (Hours 1-10): Foundation + Trajectory Mining
# WEEK 2 (Hours 11-22): GRPO Training + Evaluation

# ============================================================
# CELL 1: Environment Setup
# ============================================================

import os
import subprocess
import sys

# Install dependencies
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "fastapi", "uvicorn[standard]", "requests",
    "vllm", "bitsandbytes", "accelerate",
    "datasets", "trl>=0.9.0",
], check=True)

# Install Unsloth (for GRPO training)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "unsloth[colab-new]",
    "--extra-index-url", "https://download.pytorch.org/whl/cu121",
], check=True)

print("✅ Dependencies installed")


# ============================================================
# CELL 2: Clone / Mount Agent Code
# ============================================================

import os
REPO_PATH = "/kaggle/working/purple-agent"

# If running from a cloned repo, point to it
# Otherwise, the files should be in /kaggle/input/purple-agent/
if not os.path.exists(REPO_PATH):
    # Copy from Kaggle dataset input if available
    if os.path.exists("/kaggle/input/purple-agent"):
        subprocess.run(["cp", "-r", "/kaggle/input/purple-agent", REPO_PATH])
    else:
        os.makedirs(REPO_PATH, exist_ok=True)
        print("⚠️  Upload your purple-agent repo as a Kaggle dataset or clone it here.")

sys.path.insert(0, f"{REPO_PATH}/src")
sys.path.insert(0, f"{REPO_PATH}/scripts")
print("✅ Code path set:", REPO_PATH)


# ============================================================
# CELL 3: Check GPUs
# ============================================================

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")


# ============================================================
# CELL 4: WEEK 1 — Start vLLM Server (background)
# ============================================================

import subprocess, time, requests

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
VLLM_PORT = 8000

proc = subprocess.Popen([
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_ID,
    "--port", str(VLLM_PORT),
    "--dtype", "bfloat16",
    "--max-model-len", "8192",
    "--gpu-memory-utilization", "0.88",
    "--tensor-parallel-size", str(torch.cuda.device_count()),
    "--quantization", "bitsandbytes",
    "--trust-remote-code",
    "--disable-log-requests",
], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# Wait for it to come up
print("Waiting for vLLM to start…")
for _ in range(40):
    try:
        r = requests.get(f"http://localhost:{VLLM_PORT}/health", timeout=3)
        if r.status_code == 200:
            print(f"✅ vLLM ready at http://localhost:{VLLM_PORT}")
            break
    except:
        pass
    time.sleep(5)
    print(".", end="", flush=True)
else:
    print("\n❌ vLLM failed to start. Check GPU memory.")


# ============================================================
# CELL 5: WEEK 1 — Mine Gold Trajectories (6 hrs)
# ============================================================

from mine_trajectories import mine, parse_args
import argparse

mining_args = argparse.Namespace(
    tasks=60,                              # adjust to fit time budget
    output="/kaggle/working/gold_trajectories.jsonl",
    model_url=f"http://localhost:{VLLM_PORT}",
    max_turns=8,
)
mine(mining_args)

# Check results
import json
trajectories = []
with open(mining_args.output) as f:
    for line in f:
        trajectories.append(json.loads(line))

gold = [t for t in trajectories if t.get("found_bug")]
print(f"\n📊 Mining results: {len(gold)}/{len(trajectories)} gold trajectories")


# ============================================================
# CELL 6: WEEK 1 — Test Agent Locally (no green agent needed)
# ============================================================

from agent import PurpleAgent

agent = PurpleAgent(
    model_base_url=f"http://localhost:{VLLM_PORT}",
    model_name=MODEL_ID,
    max_turns=10,
    use_mcts=True,
    mcts_branches=2,  # use 2 branches on T4 to save time
)

test_message = {
    "problem_statement": (
        "The `add_to_set` function in `utils/collections.py` fails to handle "
        "duplicate items when the input list contains None values. "
        "Expected: None should be treated as a regular value and not cause a crash."
    ),
    "cwd": "/workspace/repo",
    "fail_to_pass": ["tests/test_collections.py::test_add_none_to_set"],
}

action = agent.respond(test_message)
print("Action:", action)


# ============================================================
# CELL 7: WEEK 2 — GRPO Training (4 hrs)
# ============================================================

# Kill vLLM first to free GPU memory for training
proc.terminate()
time.sleep(5)
print("vLLM stopped")

from grpo_train import train, parse_args as grpo_parse_args

train_args = grpo_parse_args.__wrapped__ if hasattr(grpo_parse_args, '__wrapped__') else None

# Manual args for notebook
import argparse
train_args = argparse.Namespace(
    data="/kaggle/working/gold_trajectories.jsonl",
    output="/kaggle/working/grpo_model",
    model=MODEL_ID,
    epochs=1,
    lora_r=16,
    max_seq_len=8192,
    num_generations=4,
    wandb=False,
)

train(train_args)
print("✅ GRPO training complete")


# ============================================================
# CELL 8: WEEK 2 — Restart vLLM with Fine-tuned Model
# ============================================================

FINETUNED_MODEL = "/kaggle/working/grpo_model/final"

proc2 = subprocess.Popen([
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", FINETUNED_MODEL,
    "--port", str(VLLM_PORT),
    "--dtype", "bfloat16",
    "--max-model-len", "8192",
    "--gpu-memory-utilization", "0.88",
    "--tensor-parallel-size", str(torch.cuda.device_count()),
    "--trust-remote-code",
    "--disable-log-requests",
], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# Wait
for _ in range(40):
    try:
        r = requests.get(f"http://localhost:{VLLM_PORT}/health", timeout=3)
        if r.status_code == 200:
            print(f"✅ Fine-tuned vLLM ready")
            break
    except:
        pass
    time.sleep(5)


# ============================================================
# CELL 9: WEEK 2 — Start Purple Agent Server (for AgentBeats)
# ============================================================

import threading, uvicorn
from server import app

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=9010, log_level="info")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)

# Verify
r = requests.get("http://localhost:9010/health")
print("Purple Agent health:", r.json())

r = requests.get("http://localhost:9010/.well-known/agent.json")
print("Agent card:", r.json()["name"])
print("✅ Purple Agent running on port 9010")

# For AgentBeats submission, expose port 9010 via ngrok or register Docker image
print("\n📋 Next steps:")
print("  1. Build Docker: docker build -t purple-agent .")
print("  2. Push to GHCR: docker push ghcr.io/<user>/purple-agent:latest")
print("  3. Register on agentbeats.dev")
print("  4. Submit against AgentSWE leaderboard")
