# 🟣 Purple Coding Agent

[![AgentBeats SWE-Bench Pro](https://img.shields.io/badge/AgentBeats-SWE--Bench%20Pro-purple)](https://agentbeats.dev/agentbeater/swe-bench)
[![Docker](https://img.shields.io/badge/Docker-rimodock%2Fpurple--coding--agent-blue)](https://hub.docker.com/r/rimodock/purple-coding-agent)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Stateful. Execution-Grounded. Infrastructure-Aware.**

Purple Agent is an autonomous software engineering agent optimized for the **SWE-Bench Pro** and **AIMO2026** benchmarks. In Phase 2, the architecture has transitioned from a static MCTS-guided system to a **Stateful Bash REPL** that utilizes live container execution to generate ground-truth diagnostic data.

## 🚀 The Phase 2 Pivot

Phase 1 relied on "withheld logs" and predictive reasoning. **Phase 2 (Project PhoenixSmartLite)** solves the observability problem by mounting the host Docker socket directly into the agent container. This enables:
*   **Mechanical Ground Truth**: The agent verifies its own fixes by running the test suite natively.
*   **Inference-Time Scaling**: Utilizing GRPO-styled group sampling to explore multiple diagnostic hypotheses before committing to a repair path[cite: 1].
*   **Stateful 50-Turn Loops**: A persistent Bash session that allows the model to explore, edit, and verify iteratively within a single context.
*   **Hardware Optimized**: Shift from training-heavy workflows to inference-time scaling to accommodate limited compute budgets[cite: 1].

---

## 🏗️ Architectural Pipeline

The agent operates across a strict 6-stage lifecycle coordinated via `src/server.py`[cite: 1]:

| Stage | Name | Component | Objective |
| :--- | :--- | :--- | :--- |
| **1** | **Bootstrap** | `DockerBridge` | Spin up a sibling container and mount the target repository via the Docker socket[cite: 1]. |
| **2** | **Baseline** | `TestEngine` | Identify existing failures to establish a "Relative Reward Signal"[cite: 1]. |
| **3** | **Priming** | `ICLSpecialist` | Inject framework-specific rules and AST-derived repository maps[cite: 1]. |
| **4** | **Execution** | `AgentLoop` | A 50-turn stateful REPL where the repair and reproduction happen[cite: 1]. |
| **5** | **Verification** | `TestEngine` | A mechanical gate comparing the new state against the baseline results[cite: 1]. |
| **6** | **Submission** | `server.py` | Generate the final patch and return the JSON-RPC response to the Green Agent. |

---

## 🛠️ Key Components

### 🧠 Inference-Time GRPO (`hypotheses.py`)
Instead of a single point-estimate, we use **Group Sampling** ($T=0.7$) to generate multiple diagnostic leads[cite: 1]. The agent evaluates these leads relative to the execution feedback, effectively "rewarding" strategies that reproduce the bug and "penalizing" those that return empty logs[cite: 1].

### 🗺️ Graph RAG (`ast_graph.py`)
Utilizes a lightweight parser to create a repository skeleton[cite: 1]. This provides the agent with a high-level map of classes and functions, preventing "context wandering" during the exploration of large-scale codebases[cite: 1].

### 🏗️ Docker-out-of-Docker (`docker_bridge.py`)
The "muscles" of the agent. By binding `/var/run/docker.sock`, the agent can spawn, execute commands within, and destroy isolated test environments on demand.

---

## 📦 Installation & Setup

### 1. Repository Structure
```text
├── src/
│   ├── core/         # Stateful Loop & Docker Bridge
│   ├── tools/        # AST Graph & Test Engine
│   └── prompts/      # ICL Specialist logic
├── scripts/          # Cleanup & Local Test utilities
├── amber-manifest.json5  # AgentBeats deployment config
└── pyproject.toml    # Dependency management