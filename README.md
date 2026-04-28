# 🟣 Purple Coding Agent

[![AgentBeats SWE-Bench Pro](https://img.shields.io/badge/AgentBeats-SWE--Bench%20Pro-purple)](https://agentbeats.dev/agentbeater/swe-bench)
[![Docker](https://img.shields.io/badge/Docker-rimodock%2Fpurple--coding--agent-blue)](https://hub.docker.com/r/rimodock/purple-coding-agent)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A software engineering agent built for the **AgentX–AgentBeats SWE-Bench Pro** competition. Evaluated against 100 instances of real-world GitHub issue resolution across 41 repositories.

---

## Architecture — 5-Stage Pipeline

```
Green Agent (SWE-Bench Pro)
         │  problem_statement + repo + commit
         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1 — LLM LOCALIZATION                             │
│  GitHub Tree API → LLM → top-5 file paths → fetch      │
│  (parallel asyncio.gather for file fetches)             │
├─────────────────────────────────────────────────────────┤
│  Stage 1.5 — SYNTHETIC TEST FAILURE HYPOTHESES  ★novel  │
│  Source files + bug report → LLM infers:                │
│    • Which assertion is likely failing?                  │
│    • Expected value vs. actual (buggy) value            │
│    • Root cause: exact function / condition at fault     │
│    • fix_hint: concrete code change required            │
│  Returns 2–3 ranked hypotheses (confidence-scored)       │
│  Acts as a synthetic oracle when tests are withheld     │
├─────────────────────────────────────────────────────────┤
│  Stage 2 — MCTS REPAIR (iterative, MCTS_ITERATIONS=3)  │
│  Per iteration:                                          │
│    3 parallel branches (T = 0.15 / 0.47 / 0.80)        │
│    Each branch sees: bug report + hypotheses +           │
│    source files + prior best patch (iteration > 0)      │
│  Heuristic PRM: format × relevance × completeness       │
├─────────────────────────────────────────────────────────┤
│  Stage 2.5 — PLT SELF-CONSISTENCY CHECK                 │
│  One LLM call evaluates all 3 branches against          │
│  the hypotheses:                                        │
│    • Does patch logic address root_cause?                │
│    • Is the programming technique sound?                 │
│  final_score = 0.40 × heuristic + 0.60 × PLT score     │
├─────────────────────────────────────────────────────────┤
│  UCT Selection + Backpropagate                          │
│  (non-void from iteration 2 as parent visits grow)      │
│  Early stopping if score doesn't improve               │
└─────────────────────────────────────────────────────────┘
         │  unified diff patch
         ▼
Green Agent applies patch → runs tests → pass/fail
```

---

## Key Novelty

### Hypothesis-Conditioned Repair Under Test Occlusion

SWE-Bench Pro deliberately withholds test files (`fail_to_pass`, `test_patch`, `requirements`) to prevent benchmark gaming. Every published approach accepts this blindness:

- **With execution** (SWE-agent, OpenHands): runs failing tests and reads tracebacks
- **Without execution** (Agentless, ACR): patches based on bug report alone, accepts the blindness

This agent does neither — **it refuses to accept the blindness**. Stage 1.5 uses the source files themselves as an oracle substrate: the model knows what the code *does* (readable) and what the bug report says it *should* do, and formalises that gap into structured predictions:

```json
{
  "failure_mode": "TestKernelPackageSelectionForDebugVariant: scanner returns wrong version",
  "expected_value": "kernel-debug-5.14.0-284.el9 version string",
  "actual_value": "empty string — regex fails on el9.x format",
  "root_cause": "versionRegex in scanner/redhatbase.go ~line 142 doesn't handle el9.x",
  "fix_hint": "Change regex from `el(\\d)` to `el(\\d+(?:\\.\\d+)?)`",
  "confidence": 0.95
}
```

These hypotheses then condition both patch generation (MCTS branches know *what to fix*) and patch selection (PLT checks whether each branch's logic *actually satisfies* the hypothesis).

---

## MCTS & UCT

With `MCTS_ITERATIONS = 3` and `MCTS_BRANCHES = 3`, the tree builds real visit counts across iterations:

```
UCT(j) = V̄ⱼ + √2 × √(ln N / nⱼ)
```

- Iteration 1: parent visits = 0, exploration term = 0 → greedy max
- Iteration 2: parent visits > 0, UCT becomes meaningful
- Iteration 3: full exploitation/exploration balance across 9 evaluated candidates

Between iterations, the prompt includes the **previous best patch** so the model critiques and improves rather than generating independently. This is critique-and-refine, not independent sampling.

---

## LLM Calls Per Task

| Call | Description | Tokens (max) |
|---|---|---|
| Localization | File tree → top-5 paths | 256 |
| Hypothesis synthesis | Source files → failure hypotheses | 1200 |
| MCTS branch 1,2,3 × 3 iterations | Patch generation | 2048 each |
| PLT check × 3 iterations | Branch ranking | 512 each |
| **Total** | | **~14 calls** |

At ~8–10s/call with DeepSeek-v4-flash, total task time ≈ 90–120s — well within the 300s gateway limit.

---

## Quick Start

### Prerequisites

- Docker
- OpenRouter API key (`OPENROUTER_KEY`)
- GitHub token, read-only public repos (`GITHUB_TOKEN`)

### Run Locally

```bash
# Pull and run
docker pull rimodock/purple-coding-agent:latest

docker run -p 9010:9010 \
  -e OPENROUTER_API_KEY=your_key \
  -e GITHUB_TOKEN=your_token \
  -e MODEL_NAME=deepseek/deepseek-v4-flash \
  -e MCTS_BRANCHES=3 \
  -e MCTS_ITERATIONS=3 \
  rimodock/purple-coding-agent:latest

# Health check
curl http://localhost:9010/health

# Agent card
curl http://localhost:9010/.well-known/agent-card.json
```

### Send a Test Task

```bash
curl -X POST http://localhost:9010/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "test-1",
    "method": "message/send",
    "params": {
      "message": {
        "parts": [{
          "kind": "text",
          "text": "{\"problem_statement\": \"Fix the None handling bug\", \"repo\": \"owner/repo\", \"base_commit\": \"abc123\"}"
        }]
      }
    }
  }'
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | OpenRouter API key (required) |
| `GITHUB_TOKEN` | — | GitHub token, public read-only (required) |
| `MODEL_NAME` | `deepseek/deepseek-v4-flash` | OpenRouter model ID |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | LLM API base URL |
| `MCTS_BRANCHES` | `3` | Parallel patch candidates per iteration |
| `MCTS_ITERATIONS` | `3` | Refinement rounds (UCT meaningful from round 2) |
| `USE_MCTS` | `true` | Enable/disable MCTS |
| `PORT` | `9010` | Server port |

---

## Project Structure

```
purple-coding-agent/
├── src/
│   └── server.py          # All pipeline logic (single-file architecture)
├── docker/                # Docker build configs
├── scripts/               # Utility scripts
├── tests/                 # Test suite
├── Dockerfile
├── docker-compose.yml
├── amber-manifest.json5   # AgentBeats manifest
├── requirements.txt
└── README.md
```

`server.py` contains the full pipeline as a single cohesive file:
- `LLMClient` — OpenRouter client with `reasoning_content` fallback for thinking models
- `llm_localize()` — Stage 1 file localization
- `llm_synthesize_hypotheses()` — Stage 1.5 synthetic oracle construction
- `llm_plt_consistency_check()` — Stage 2.5 PLT self-consistency scoring
- `ProgrammablePRM` — Heuristic scorer (format / relevance / completeness)
- `MCTSEngine` — UCT tree with backpropagation
- `PurpleAgent` — Main orchestrator
- FastAPI A2A server

---

## Build & Push

```bash
# Build
docker build -t rimodock/purple-coding-agent:latest .

# Push
docker push rimodock/purple-coding-agent:latest
```

GitHub Actions automatically builds and pushes on every push to `main`.

---

## AgentBeats Submission

1. Go to [agentbeats.dev](https://agentbeats.dev) → Quick Submit
2. Docker image: `rimodock/purple-coding-agent:latest`
3. Required secrets:
   - `OPENROUTER_KEY` → your OpenRouter API key
   - `GITHUB_TOKEN` → GitHub PAT (public repos, read-only)

---

## Roadmap

- [x] Stage 1: LLM localization via GitHub Tree API
- [x] Stage 1.5: Synthetic test failure hypothesis generation
- [x] Stage 2: MCTS repair with parallel branch generation
- [x] Stage 2.5: PLT self-consistency check
- [x] Iterative refinement with prior-patch conditioning (UCT-meaningful)
- [x] DeepSeek-v4-flash integration with reasoning_content fallback
- [ ] Phase 2: Multi-turn bash exploration with local Docker execution
- [ ] Graph RAG: Call-graph traversal for hypothesis grounding
- [ ] GRPO fine-tuning on (hypothesis, patch, pass/fail) triples

---

## Competition

- **Platform:** [AgentBeats](https://agentbeats.dev/agentbeater/swe-bench)
- **Benchmark:** SWE-Bench Pro — 100 instances, 41 repositories
- **Protocol:** A2A (Google Agent-to-Agent)
- **Green Agent:** Single-turn — sends problem statement, awaits patch

---

## License

MIT