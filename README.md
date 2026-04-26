# 🟣 Phoenix Agent — AgentBeats Phase 2

[![CI](https://github.com/YOUR_USERNAME/purple-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/purple-agent/actions)
[![AgentBeats](https://img.shields.io/badge/AgentBeats-Phase%202-purple)](https://agentbeats.dev)

An **MCTS-guided software engineering agent** built for the [AgentX–AgentBeats](https://rdi.berkeley.edu/agentx-agentbeats) Phase 2 competition (Coding Agent track). Evaluated against **AgentSWE** (SWE-bench Pro).

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                    PURPLE AGENT                             │
│                                                            │
│  ┌──────────────┐    ┌────────────────┐    ┌───────────┐  │
│  │  A2A Server  │───▶│   MCTS Engine  │───▶│    PRM    │  │
│  │  (FastAPI)   │    │  (UCT select)  │    │ (scorer)  │  │
│  └──────────────┘    └────────────────┘    └───────────┘  │
│          │                   │                             │
│          ▼                   ▼                             │
│  ┌──────────────┐    ┌────────────────┐                   │
│  │  State Mgr   │    │   LLM Client   │                   │
│  │  (per node)  │    │  (vLLM/HF)     │                   │
│  └──────────────┘    └────────────────┘                   │
│                              │                             │
└──────────────────────────────│─────────────────────────────┘
                               ▼
                  ┌────────────────────────┐
                  │   vLLM / Unsloth       │
                  │  DeepSeek V3.2   │
                  └────────────────────────┘
```

### Key Components

| Component | File | Purpose |
|---|---|---|
| A2A Server | `src/server.py` | FastAPI HTTP server implementing the A2A protocol |
| Agent Core | `src/agent.py` | Orchestrates MCTS + LLM reasoning loop |
| MCTS Engine | `src/mcts.py` | UCT-based action selection across tree branches |
| PRM | `src/prm.py` | Programmable Process Reward Model (3-layer scoring) |
| State Manager | `src/state.py` | Per-branch state (files, discoveries, patch) |
| LLM Client | `src/llm_client.py` | OpenAI-compatible client + HuggingFace fallback |
| Prompts | `src/prompts.py` | XML-structured prompts for TIR format |

---

## Agent Protocol

This agent implements the [A2A Protocol](https://a2a-protocol.org/) and is compatible with **AgentSWE** (green agent for SWE-bench Verified).

### Endpoints

```
GET  /.well-known/agent.json   # Agent card
GET  /health                   # Health check
POST /                         # Handle A2A task message
```

### Interaction Modes

The agent supports all three AgentSWE modes:

| Mode | Usage |
|---|---|
| `bash` | Read-only codebase exploration (`find`, `grep`, `cat`, `pytest`) |
| `debug` | Ephemeral writes to test hypotheses |
| `patch` | Final unified diff submission |

### Output Format (TIR)

The LLM is prompted to output structured XML:

```xml
<thought>
Step-by-step analysis. References specific file names and line numbers.
Verifies reasoning before acting.
</thought>
<command>bash</command>
<content>
grep -rn "add_to_set" utils/ | head -20
</content>
```

---

## Quick Start

### Prerequisites

- Docker + Docker Compose
- NVIDIA GPU (for vLLM; 16GB+ VRAM recommended)
- `HF_TOKEN` environment variable (for gated models)

### Run Locally

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/purple-agent
cd purple-agent

# 2. Start agent + vLLM
HF_TOKEN=your_token docker compose up

# 3. Verify
curl http://localhost:9010/health
curl http://localhost:9010/.well-known/agent.json

# 4. Send a test task
curl -X POST http://localhost:9010/ \
  -H "Content-Type: application/json" \
  -d '{
    "problem_statement": "Fix the None handling bug in utils/collections.py",
    "cwd": "/workspace/repo",
    "fail_to_pass": ["tests/test_collections.py::test_none_handling"]
  }'
```

### Run Tests (no GPU needed)

```bash
pip install -e ".[dev]"
pip install httpx pytest-asyncio
pytest tests/ -v
```

---

## Kaggle 2×T4 Training Guide (22 GPU Hours)

### Week 1 — Foundation & Trajectory Mining (10 Hours)

| Hours | Task | GPU Usage |
|---|---|---|
| 1–3 | Start vLLM server + test A2A endpoints | 1 hr |
| 4–8 | Mine gold trajectories from 60 SWE-bench tasks | 6 hrs |
| 9–10 | Validate trajectories + PRM calibration | 2 hrs |

```python
# In Kaggle notebook — start vLLM
exec(open("scripts/kaggle_notebook.py").read())
```

### Week 2 — GRPO Training & Sprint Launch (12 Hours)

| Hours | Task | GPU Usage |
|---|---|---|
| 11–15 | GRPO fine-tuning with Unsloth MoE kernels | 4 hrs |
| 16–20 | Re-run full eval with tuned model + MCTS | 4 hrs |
| 21–22 | Build Docker image, register on AgentBeats | 1 hr |

```bash
# GRPO training (from Kaggle notebook)
python scripts/grpo_train.py \
  --data /kaggle/working/gold_trajectories.jsonl \
  --output /kaggle/working/grpo_model \
  --epochs 1 \
  --lora-r 16
```

---

## Docker Submission

### Build

```bash
docker build -t purple-agent:latest .
```

### Push to GHCR (auto via GitHub Actions on push to main)

```bash
# Manual push
docker tag purple-agent:latest ghcr.io/YOUR_USERNAME/purple-agent:latest
docker push ghcr.io/YOUR_USERNAME/purple-agent:latest
```

### Register on AgentBeats

1. Go to [agentbeats.dev/register-agent](https://agentbeats.dev/register-agent)
2. Select **Purple**
3. Set Docker image: `ghcr.io/YOUR_USERNAME/purple-agent:latest`
4. Add required env vars: `LLM_BASE_URL`, `MODEL_NAME`
5. Submit against the **AgentSWE** leaderboard

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://vllm:8000` | vLLM/OpenAI-compatible endpoint |
| `MODEL_NAME` | `deepseek/deepseek-chat-v3-0324` | Model ID |
| `MAX_TURNS` | `15` | Max turns per task |
| `MCTS_BRANCHES` | `3` | Candidate actions per MCTS step |
| `TEMPERATURE` | `0.6` | LLM sampling temperature |
| `USE_MCTS` | `true` | Enable/disable MCTS (disable for speed) |
| `PORT` | `9010` | Purple agent server port |

---

## MCTS Search Strategy

The agent uses **UCT (Upper Confidence Trees)** to balance exploration vs exploitation across candidate actions:

$$UCT(j) = \bar{V}_j + c\sqrt{\frac{\ln N}{n_j}}$$

Where:
- $\bar{V}_j$ = average PRM reward of node $j$
- $N$ = parent visit count
- $n_j$ = node visit count  
- $c = \sqrt{2}$ (exploration constant)

At each turn:
1. LLM samples **3 candidate actions** (bash/debug/patch)
2. **Programmable PRM** scores each statically (format + relevance)
3. **UCT** selects the best branch to execute
4. After observation, reward is backpropagated through the tree

---

## Reward Model (PRM)

Three-layer scoring ∈ [0, 1]:

| Layer | Weight | Signal |
|---|---|---|
| Format ($R_f$) | 20% | Valid XML structure, correct command type |
| Relevance ($R_l$) | 35% | Content mentions problem-relevant tokens/files |
| Execution ($R_e$) | 45% | Test pass/fail, file discovery, error absence |

---

## GRPO Training Details

Fine-tuning uses **Unsloth's MoE Triton Kernels** for 7× speed and 35% less VRAM:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek/deepseek-chat-v3-0324",
    max_seq_length=8192,
    load_in_4bit=True,
)
```

Reward functions:
- `reward_format` — XML structure validation
- `reward_logic` — TIR file-path grounding
- `reward_patch` — Diff format correctness

---

## Project Structure

```
purple-agent/
├── src/
│   ├── server.py          # A2A FastAPI server
│   ├── agent.py           # Core orchestrator
│   ├── mcts.py            # MCTS engine (UCT)
│   ├── prm.py             # Programmable reward model
│   ├── state.py           # Node state manager
│   ├── llm_client.py      # vLLM / HuggingFace client
│   └── prompts.py         # Structured XML prompts
├── scripts/
│   ├── grpo_train.py      # GRPO fine-tuning (Week 2)
│   ├── mine_trajectories.py # Gold trajectory collection (Week 1)
│   ├── kaggle_notebook.py # Full Kaggle workflow
│   └── start_vllm.sh      # vLLM server launcher
├── tests/
│   └── test_agent.py      # Unit + integration tests
├── .github/workflows/
│   └── ci.yml             # CI + Docker build/push
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Competition Details

- **Competition:** [AgentX–AgentBeats Phase 2](https://rdi.berkeley.edu/agentx-agentbeats)
- **Track:** Coding Agent (Sprint 3, Apr 13 – May 3)
- **Green Agent:** [AgentSWE](https://agentbeats.dev/agentbeater/swe-bench) (SWE-bench Pro)
- **Metric:** Resolved Rate (pass@1) + Token Efficiency

---

## License

MIT
