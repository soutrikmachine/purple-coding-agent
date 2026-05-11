# 🟣 Purple Coding Agent

[![AgentBeats SWE-Bench Pro](https://img.shields.io/badge/AgentBeats-SWE--Bench%20Pro-purple)](https://agentbeats.dev/agentbeater/swe-bench)
[![Docker](https://img.shields.io/badge/Docker-rimodock%2Fpurple--coding--agent-blue)](https://hub.docker.com/r/rimodock/purple-coding-agent)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Project PhoenixSmartLite: Hypothesis-Conditioned Stateful REPL Agent for SWE-bench Pro**

---

## Abstract

Purple Agent (Phase 2) is an autonomous software engineering agent built for the **AgentX–AgentBeats SWE-Bench Pro** competition. The architecture evolved from a static MCTS-guided patch generator (Phase 1) to a **Stateful Bash REPL** that runs code inside live Docker sibling containers, enabling real test execution feedback rather than inferred patch quality.

The key insight from Phase 1: *sophisticated static analysis has a hard ceiling*. An agent can correctly identify the root cause, generate a plausible patch, and score it highly — and still fail every test, because without executing the code there is no ground truth. Phase 2 eliminates this blindness by mounting the host Docker socket directly into the agent container (`/var/run/docker.sock`), enabling the agent to spawn isolated SWE-bench repository environments and run their test suites natively.

Phase 2 retains the novel **Synthetic Test Failure Hypothesis** engine from Phase 1 — repurposed from patch-time oracle to exploration-time guide. Rather than compensating for withheld test files (Phase 1's constraint), hypotheses now tell the agent *where to look first*, saving 5–8 bash turns of blind `grep` before locating the bug site.

The pipeline uses **Gemini Flash** (via OpenRouter) as its primary reasoning model, chosen for REPL-optimised behaviour: it responds reliably in structured XML format, thinks automatically between turns, and routes its chain-of-thought into per-turn NOTES.txt entries that persist across the pruned context window — giving the agent external long-term memory without token explosion.

---

## Architecture — 6-Stage Pipeline

```
Green Agent (SWE-Bench Pro)
    │  problem_statement + docker_image + repo + base_commit
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 — CONTAINER BOOTSTRAP                              │
│  Pull SWE-bench Docker image → git checkout base_commit     │
│  Auto-provision: Redis / MongoDB / PostgreSQL if detected   │
├─────────────────────────────────────────────────────────────┤
│  Stage 1.5 — HYPOTHESIS SYNTHESIS (Using GRPO)  ★                        │
│  find /repo_dir → file tree → LLM infers:                  │
│    • Which file/function is the likely bug site?            │
│    • What bash command reproduces the failure?              │
│    • What reasoning supports each hypothesis?               │
│  2–3 ranked leads injected into turn-1 context              │
├─────────────────────────────────────────────────────────────┤
│  Stage 2 — TEST COMMAND DISCOVERY (lightweight, ~5s)        │
│  Probes pytest.ini / go.mod / package.json / Cargo.toml     │
│  Writes run_script.sh into the container workspace          │
│  No test execution at this stage (avoids 3-min stalls)      │
├─────────────────────────────────────────────────────────────┤
│  Stage 3 — ICL INJECTION                                    │
│  Domain detection: Django / pytest / Go / Node / general    │
│  Framework-specific rules + few-shot REPL examples          │
├─────────────────────────────────────────────────────────────┤
│  Stage 4 — STATEFUL BASH REPL  (up to MAX_TURNS=20)        │
│  3 tools injected into /workspace:                          │
│    • edit_file.py   — safe file editor (no sed)             │
│    • ast_search.py  — grep-based function/class locator     │
│    • run_script.sh  — discovered test command wrapper       │
│  NOTES.txt auto-updated by framework after every turn       │
│  Gemini reasoning_content written to NOTES.txt (per-turn    │
│    thinking memory without polluting message context)        │
│  Context window: system + task + rolling last-8 messages    │
│  Urgency warnings at turns MAX_TURNS-6 and MAX_TURNS-2      │
│  Force-extract git diff at MAX_TURNS                        │
├─────────────────────────────────────────────────────────────┤
│  Stage 5 — MECHANICAL TEST GATE                             │
│  Run full test suite in container                           │
│  Baseline-aware: only NEW regressions count as failures     │
│  PASS → accept patch     FAIL → enter Stage 6              │
├─────────────────────────────────────────────────────────────┤
│  Stage 6 — QA FIX PHASE  (up to 2 retries × 3 turns)      │
│  Inject specific failing test IDs                           │
│  Agent runs targeted tests (pytest path::name -x)           │
│  Re-runs gate after each repair burst                       │
└─────────────────────────────────────────────────────────────┘
    │  git diff HEAD  (always extracted, even on gate failure)
    ▼
Green Agent applies patch → runs tests → pass/fail
```

---

## Key Design Decisions

### Why the hypothesis engine survives from Phase 1

In Phase 1, hypotheses tried to *replace* missing test signal. In Phase 2, they *accelerate* exploration. An agent starting a 20-turn budget cold will spend turns 1–6 on blind file tree navigation. With a hypothesis that says "bug is in `scanner/redhatbase.go`, run `go test -run TestKernelDebug ./...`", the agent can validate or discard that lead in turn 1 and go straight to editing by turn 3. This is the tutor analogy: the agent enters the bash loop as a domain-informed expert, not a generalist.

### Why NOTES.txt as external memory

The rolling 8-message context window (kept to control token costs) means the agent loses its exploration history every 4 turns. NOTES.txt is written by the **framework** after every bash execution — not by the agent — so it's reliable regardless of model compliance. The agent is instructed to read it when lost, giving it O(1) access to its full turn history at the cost of one bash command.

### Why Gemini Flash for REPL

Gemini 2.5/3 Flash:
- Never returns `content=null` (unlike DeepSeek thinking models which broke the REPL parser)
- Thinks automatically between turns (chain-of-thought written to NOTES.txt as a side channel)
- Responds in structured XML when explicitly instructed
- Routes directly to Google AI Platform via OpenRouter (no Parasail/NovitaAI queueing latency)

---

## Timing Budget

```
Global timeout:               260s  (asyncio.wait_for — 40s under 300s gateway)
├─ Container bootstrap:        40s  (pull → start → git checkout → services)
├─ Hypothesis synthesis:       10s  (asyncio.wait_for cap, degrades to [])
├─ Test command discovery:      5s  (probe only, no execution)
├─ REPL loop (20 turns × 11s):220s  ← most time
│   Observation cap: 1500 chars (head+tail, prevents token explosion)
├─ Test gate + QA:             20s
└─ Patch extraction:            5s

Periodic git diff snapshot every 4 turns → /tmp/purple_patch.diff
If global timeout fires: reads /tmp/purple_patch.diff as best-effort patch
```

---

## Estimated Cost per Run (100 tasks)

With `google/gemini-3-flash-preview` via OpenRouter ($1.34/M input · $3/M output):

| Component | Tokens/task | Cost/task |
|---|---|---|
| Input (rolling context, 16 avg turns) | ~55,000 | $0.074 |
| Output (response) | ~6,500 | $0.020 |
| Gemini thinking (auto, ~800 tok/turn) | ~13,000 | $0.039 |
| Hypothesis pre-flight | ~3,500 | $0.005 |
| **Total** | **~78,000** | **~$0.138** |

**100 tasks: ~$10–14** (less for tasks solved early, capped at timeout for hard ones)

---

## Project Structure

```
purple-coding-agent/
├── src/
│   ├── server.py              # FastAPI A2A server + global timeout wrapper
│   ├── core/
│   │   ├── agent_loop.py      # Stage 4 & 6: 20-turn REPL engine
│   │   ├── docker_bridge.py   # Stage 1: DooD container lifecycle
│   │   └── llm_client.py      # Gemini/DeepSeek client, XML parser, fence stripper
│   ├── tools/
│   │   ├── test_engine.py     # Stage 2, 5: test discovery + mechanical gate
│   │   ├── hypotheses.py      # Stage 1.5: GSRM group sampling (T=0.7, 2–3 leads)
│   │   └── ast_graph.py       # Graph RAG: tree-sitter skeleton (Python + JS)
│   └── prompts/
│       └── icl_specialist.py  # Stage 3: domain detection + ICL injection
├── tests/
│   └── test_core.py           # 28 unit tests across all components
├── scripts/                   # Utility scripts
├── Dockerfile                 # python:3.11-slim + Docker CLI (DooD requirement)
├── docker-compose.yml
├── amber-manifest.json5       # AgentBeats deployment config (Docker socket mount)
└── pyproject.toml
```

---

## Environment Variables

All tunable without rebuilding the image:

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | OpenRouter API key (required) |
| `MODEL_NAME` | `google/gemini-3-flash-preview` | OpenRouter model slug — verify at openrouter.ai/models |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | LLM endpoint |
| `MAX_TURNS` | `20` | REPL turn budget (hard cap — force-submits at this turn) |
| `MAX_OBS_CHARS` | `1500` | Observation output cap (head+tail, prevents token explosion) |
| `CONTEXT_KEEP` | `8` | Rolling message window (last N messages kept after pruning) |
| `TASK_TIMEOUT_S` | `260` | Global asyncio timeout — must stay under 300s gateway limit |
| `PORT` | `9022` | Server port |

---

## Quick Start

```bash
# Run locally
docker run -p 9022:9022 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e OPENROUTER_API_KEY=your_key \
  -e MODEL_NAME=google/gemini-3-flash-preview \
  rimodock/purple-coding-agent:latest

# Health check
curl http://localhost:9022/health

# Agent card
curl http://localhost:9022/.well-known/agent-card.json
```

Or with docker-compose (includes optional local vLLM profile):

```bash
OPENROUTER_API_KEY=your_key docker compose up
```

## Run Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## AgentBeats Submission

1. Go to [agentbeats.dev](https://agentbeats.dev) → SWE-bench → Quick Submit
2. Docker image: `rimodock/purple-coding-agent:latest`
3. Required secrets: `OPENROUTER_API_KEY`

The `amber-manifest.json5` handles Docker socket mounting automatically via `"from": "framework.docker"`.

---

## Phase Roadmap

- [x] Phase 1: LLM localization → synthetic hypothesis synthesis → MCTS repair → PLT self-consistency
- [x] Phase 2: Docker-out-of-Docker REPL → mechanical test gate → hypothesis-guided exploration
- [ ] Phase 3: Graph RAG call-graph traversal (import-following) for deeper hypothesis grounding
- [ ] Phase 3: GRPO fine-tuning on (hypothesis, bash_trace, pass/fail) triples from Phase 2 runs

---

## Competition

- **Platform:** [AgentBeats](https://agentbeats.dev/agentbeater/swe-bench)
- **Benchmark:** SWE-Bench Pro — 100 instances, 41 repositories
- **Protocol:** A2A (Google Agent-to-Agent)
- **Green Agent:** Single-turn per task — sends problem + Docker image, awaits patch

---

## License

MIT