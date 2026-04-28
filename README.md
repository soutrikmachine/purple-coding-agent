# 🟣 Purple Coding Agent — AgentBeats Phase 2

[![AgentBeats](https://img.shields.io/badge/AgentBeats-Phase%202-purple)](https://agentbeats.dev)
[![Competition](https://img.shields.io/badge/AgentX-SWE--bench%20Pro-blue)](https://rdi.berkeley.edu/agentx-agentbeats)
[![Docker](https://img.shields.io/badge/Docker-rimodock%2Fpurple--coding--agent-blue)](https://hub.docker.com/r/rimodock/purple-coding-agent)

A **multi-turn software engineering agent** built for the [AgentX–AgentBeats](https://rdi.berkeley.edu/agentx-agentbeats) Phase 2 competition (Coding Agent track), evaluated against **AgentSWE** on **SWE-bench Pro**.

---

## Architecture Overview

The agent uses a **3-stage pipeline** that combines static analysis, interactive bash exploration, and parallel patch generation:

```
┌─────────────────────────────────────────────────────────────────┐
│                     PURPLE CODING AGENT (v3)                    │
│                                                                 │
│  ┌─────────────┐                                                │
│  │  A2A Server │  FastAPI — handles JSON-RPC multi-turn msgs    │
│  │  (FastAPI)  │                                                │
│  └──────┬──────┘                                                │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   STAGE 1 — LOCALIZATION                │   │
│  │  GitHub Tree API → filter 300 source files →            │   │
│  │  LLM reasons: "which files contain the bug?" →          │   │
│  │  Returns JSON array of file paths                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              STAGE 2 — BASH EXPLORATION                 │   │
│  │  Turns 1 to MAX_TURNS-1:                                │   │
│  │  • LLM decides next bash command (grep, cat, pytest)    │   │
│  │  • Green agent executes in real repo Docker container   │   │
│  │  • stdout/stderr returned in next A2A turn              │   │
│  │  • Full conversation history accumulated                │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                STAGE 3 — MCTS REPAIR                    │   │
│  │  Final turn:                                            │   │
│  │  • 3 parallel patch candidates (asyncio.gather)         │   │
│  │  • Temperature diversity: 0.15 → 0.48 → 0.80           │   │
│  │  • PRM scores each: format + relevance + completeness   │   │
│  │  • UCT-based MCTS selects highest-scoring valid patch   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Multi-Turn?

SWE-bench Pro deliberately withholds test names, requirements, and interface specs. The green agent only sends:

```json
{
  "problem_statement": "...",
  "repo": "ansible/ansible",
  "base_commit": "abc123..."
}
```

Without seeing what the failing test actually checks, single-turn blind patching generates valid diffs that fix the wrong code. Multi-turn bash exploration solves this:

```
Turn 1 → bash: "cat -n lib/config/cluster.go | head -100"
Turn 2 ← real file content with real line numbers
Turn 3 → bash: "go test ./lib/... -run TestClusterConfig --tb=short"
Turn 4 ← ACTUAL test failure: "expected true, got false at line 67"
Turn 5 → bash: "grep -n 'isValid' lib/config/cluster.go"
Turn 6 → patch: informed diff at exact line with exact context
```

The green agent executes our bash commands in the repo's Docker container and returns stdout/stderr — we get interactive shell access without managing Docker ourselves.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| LLM localization over keyword search | Semantics beat syntax — LLM understands "caching bug in cluster config → look in `lib/services/`" |
| Bash exploration before patching | Real stack traces beat static GitHub snapshots |
| `asyncio.gather` for MCTS branches | All 3 repair candidates fire simultaneously — same wall-clock time as 1 |
| `contextId` as session key | Persists across ALL turns; `instance_id` only appears in turn 1 |
| No `"reasoning"` param in Gemma 4 payload | Causes `content: null` crash; Gemma 4 reasons naturally via `<think>` blocks |
| Strip `<think>` before patch extraction | Prevents reasoning tokens from corrupting the unified diff |

---

## Repository Structure

```
purple-coding-agent/
├── src/
│   └── server.py            # Complete single-file implementation (v3)
│                            # Contains: A2A server, localization, bash exploration,
│                            # MCTS engine, PRM, LLM client, session management
├── Dockerfile               # Single-stage, linux/amd64
├── requirements.txt         # fastapi, uvicorn, requests, httpx
├── amber-manifest.json5     # AgentBeats deployment config
└── README.md
```

Everything is in `src/server.py` — no separate modules needed.

---

## Agent Protocol (A2A)

Implements the [A2A Protocol](https://a2a-protocol.org/) spec:

### Endpoints

```
GET  /.well-known/agent-card.json   # Agent capability declaration
GET  /.well-known/agent.json        # Compatibility alias
GET  /health                        # Health check
POST /                              # Handle A2A JSON-RPC message (all turns)
```

### Action Types

| Action | When | Content |
|---|---|---|
| `bash` | Exploration turns 1 to N-1 | Shell command executed in repo container |
| `patch` | Final turn | Unified git diff starting with `diff --git` |

### Exploration Prompt Format

The LLM is instructed to respond in XML during exploration:

```xml
<thought>
I can see the cluster config caching is done in lib/config/cluster.go.
The isValidationPending check on line 67 uses stale cache — need to
force refresh on session renewal.
</thought>
<action>bash</action>
<content>
grep -n "isValidationPending\|cache\|refresh" lib/config/cluster.go | head -30
</content>
```

---

## Session Lifecycle

```
POST / (turn 1)  → extract contextId → create session
                 → Stage 1: fetch tree → LLM localize → fetch files
                 → return bash action (explore located files)

POST / (turn 2)  → same contextId → record observation (stdout/stderr)
                 → LLM decides next bash command
                 → return bash action

POST / (turns 3-5) → accumulate exploration history

POST / (turn 6)  → MAX_TURNS reached
                 → Stage 3: MCTS repair with full context
                 → return patch action
```

---

## Quick Start

### Prerequisites

- Docker
- OpenRouter API key (for Gemma 4 31B)
- GitHub personal access token (for repo tree/file fetching)

### Run Locally

```bash
# Clone
git clone https://github.com/soutrikmachine/purple-coding-agent
cd purple-coding-agent

# Build
docker buildx build --platform linux/amd64 -t purple-coding-agent:local .

# Run
docker run -p 9010:9010 \
  -e OPENROUTER_API_KEY=your_key \
  -e GITHUB_TOKEN=your_token \
  purple-coding-agent:local

# Verify
curl http://localhost:9010/health
curl http://localhost:9010/.well-known/agent-card.json
```

### Deploy to AgentBeats

```bash
# Build and push (M1/M2 Mac cross-compile)
docker buildx build --no-cache --platform linux/amd64 \
  -t rimodock/purple-coding-agent:latest --push .
```

Then Quick Submit on [agentbeats.dev](https://agentbeats.dev) with secrets:
- `OPENROUTER_KEY` → your OpenRouter API key
- `GITHUB_TOKEN` → your GitHub personal access token

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible LLM endpoint |
| `MODEL_NAME` | `deepseek/deepseek-v4-flash` | Model ID on OpenRouter |
| `OPENROUTER_API_KEY` | — | OpenRouter API key (set as secret) |
| `GITHUB_TOKEN` | — | GitHub PAT for tree/file API (set as secret) |
| `PORT` | `9010` | Agent server port |
| `MAX_TURNS` | `6` | Bash exploration turns before forcing patch |
| `MCTS_BRANCHES` | `3` | Parallel patch candidates in Stage 3 |
| `USE_MCTS` | `true` | Enable MCTS selection (false = greedy) |

---

## MCTS Patch Selection

On the final turn, 3 patch candidates are generated simultaneously via `asyncio.gather` with temperature diversity:

$$T_i = 0.15 + \frac{0.65}{N-1} \cdot i \quad \text{for } i \in \{0, 1, 2\}$$

Giving temperatures **[0.15, 0.48, 0.80]** — conservative, balanced, creative.

Each candidate is scored by the **Programmable Reward Model (PRM)**:

| Layer | Weight | Signal |
|---|---|---|
| Format | 35% | Valid `diff --git` structure, `@@` hunks, `+`/`-` lines |
| Relevance | 35% | Overlap between patch tokens and problem statement |
| Completeness | 30% | Non-empty, non-truncated diff |

UCT-based MCTS selects the highest-scoring candidate and backpropagates the reward.

---

## Benchmark Context

| Benchmark | Our Agent | SOTA |
|---|---|---|
| SWE-bench Pro | 0.0% (work in progress) | 58.6% (Kimi-K2.6) |
| SWE-bench Verified | not yet evaluated | ~70%+ |

SWE-bench Pro is the hardest coding benchmark — average gold patch is **107 lines across 4.1 files**, and every task deliberately excludes trivially solvable bugs. Even GPT-5 scores ~23%.

Current status: patches apply cleanly (`pass_to_pass_ok: True`) but fix the wrong code (`fail_to_pass_ok: False`). Multi-turn exploration is the structural fix for this.

---

## Development History

| Version | Architecture | Pass Rate |
|---|---|---|
| v1 | Single-turn blind patching (keyword file search) | 1% (lucky) |
| v2 | Two-stage: LLM localization + MCTS repair | 0% (wrong code) |
| v3 | Three-stage: localization + bash exploration + MCTS | TBD |

Key lessons learned:
- `pass_to_pass_ok: True` on all tasks — patch format and git apply are correct
- `fail_to_pass_ok: False` on all tasks — model patches adjacent code, not the bug
- Green agent withholds `fail_to_pass`, `requirements`, `interface`, `test_patch`
- Keyword file search fetches semantically wrong files
- LLM localization (semantic reasoning over repo tree) fetches the right files
- Without interactive execution, even correct files don't tell you what line to change

---

## Roadmap

- [x] A2A protocol (JSON-RPC, contextId session, correct response format)
- [x] GitHub file fetching (raw.githubusercontent.com + Tree API)
- [x] LLM-based semantic localization
- [x] Async MCTS with PRM scoring
- [x] Multi-turn bash exploration loop
- [ ] Validate multi-turn pass rate improvement
- [ ] Switch to DeepSeek V4 Pro for serious leaderboard submission
- [ ] Implement GRPO fine-tuning on SWE-bench Lite trajectories
- [ ] Evaluate on SWE-bench Verified, SWE-bench Lite, Terminal Bench

---

## Competition

- **Competition:** [AgentX–AgentBeats Phase 2](https://rdi.berkeley.edu/agentx-agentbeats)
- **Track:** Coding Agent — Sprint 3 (Apr 13 – May 3, 2026)
- **Evaluation:** SWE-bench Pro (100 tasks, 20 shards)
- **Metric:** Pass rate — `fail_to_pass_ok` across all instances
- **Leaderboard:** [agentbeats.dev/agentbeater/swe-bench](https://agentbeats.dev/agentbeater/swe-bench)

---

## License

MIT