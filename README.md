# 🟣 Purple Coding Agent

[![AgentBeats SWE-Bench Pro](https://img.shields.io/badge/AgentBeats-SWE--Bench%20Pro-purple)](https://agentbeats.dev/agentbeater/swe-bench)
[![Docker](https://img.shields.io/badge/Docker-rimodock%2Fpurple--coding--agent-blue)](https://hub.docker.com/r/rimodock/purple-coding-agent)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Project PhoenixSmartLite: An Inference-Scaled, Infrastructure-Resilient Autonomous Agent for SWE-bench Pro**

Purple Agent is an autonomous software engineering agent optimized for the **SWE-Bench Pro** benchmarks. In Phase 2, the architecture has transitioned from a static MCTS-guided system to a **Stateful Bash REPL** that utilizes live container execution to generate ground-truth diagnostic data.

## Abstract

Driven by severe hardware compute constraints and the notoriously obfuscated nature of evaluation platforms like SWE-bench Pro, the development of the Purple Coding Agent (Phase 2) represents a fundamental shift from resource-heavy supervised fine-tuning to inference-time scaling. Standard autonomous agents frequently fail not due to a lack of coding logic, but because they are derailed by "dirty" repository states, missing background databases, and blind codebase navigation. The primary motivation behind this architecture was to decouple infrastructure traps from cognitive coding tasks, maximizing the capabilities of high-efficiency reasoning models (such as DeepSeek-v4) by completely eliminating DevOps overhead.

To achieve this, the Purple Agent employs a single, flat stateful Bash REPL loop enveloped in a highly sophisticated, multi-stage pipeline. The architecture replaces standard file-tree string matching with an Abstract Syntax Tree (AST) Graph RAG, providing the model with a deterministic, surgical map of repository structures (classes, methods, and imports). Rather than relying on a blind single-loop execution, the agent utilizes principles of Group Relative Policy Optimization (GRPO) to generate, rank, and score multiple repair hypotheses prior to execution, mitigating the "rabbit hole" effect common in complex repository debugging. Execution is managed by an ironclad Docker-out-of-Docker (DooD) bridge that bypasses standard proxy streaming limitations via host-side subprocess execution. This infrastructure layer also acts as an "Auto-DevOps" orchestrator, heuristically scanning configurations to silently provision required background services (e.g., Redis, MongoDB, PostgreSQL) before the LLM assumes control.

The key capabilities of the Purple Agent center around its highly optimized feedback loops. It introduces a "Smarter Test Gate" that executes a baseline pre-check to capture pre-existing repository failures. By applying strict set-math to test execution IDs, the mechanical gate filters out environmental noise, ensuring the agent is evaluated—and rewarded—exclusively on newly introduced regressions. When regressions do occur, the agent abandons slow, broad test suites in favor of a rapid-response QA micro-loop. This phase isolates the specific failing tests and forces the LLM into a targeted, fast-iteration repair burst. Finally, the patch extraction layer safely stages untracked code and filters binary artifacts, guaranteeing an evaluation-ready unified diff. By synthesizing defensive, elite-level infrastructure with inference-scaled cognitive planning, the Purple Agent achieves state-of-the-art repository repair capabilities within strict computational and turn-limit budgets.

---

## 🚀 The Phase 2 Pivot

Phase 1 relied on "withheld logs" and predictive reasoning. **Phase 2 (Project PhoenixSmartLite)** solves the observability problem by mounting the host Docker socket directly into the agent container. This enables:
*   **Mechanical Ground Truth**: The agent verifies its own fixes by running the test suite natively.
*   **Inference-Time Scaling**: Utilizing GRPO-styled group sampling to explore multiple diagnostic hypotheses before committing to a repair path
*   **Stateful 50-Turn Loops**: A persistent Bash session that allows the model to explore, edit, and verify iteratively within a single context.
*   **Hardware Optimized**: Shift from training-heavy workflows to inference-time scaling to accommodate limited compute budgets.

---

## 🏗️ Architectural Pipeline

The agent operates across a strict 6-stage lifecycle coordinated via `src/server.py`[cite: 1]:

| Stage | Name | Component | Objective |
| :--- | :--- | :--- | :--- |
| **1** | **Bootstrap** | `DockerBridge` | Spin up a sibling container and mount the target repository via the Docker socket |
| **2** | **Baseline** | `TestEngine` | Identify existing failures to establish a "Relative Reward Signal" |
| **3** | **Priming** | `ICLSpecialist` | Inject framework-specific rules and AST-derived repository maps |
| **4** | **Execution** | `AgentLoop` | A 50-turn stateful REPL where the repair and reproduction happen |
| **5** | **Verification** | `TestEngine` | A mechanical gate comparing the new state against the baseline results |
| **6** | **Submission** | `server.py` | Generate the final patch and return the JSON-RPC response to the Green Agent. |

---

## 🛠️ Key Components

### 🧠 Inference-Time GRPO (`hypotheses.py`)
Instead of a single point-estimate, we use **Group Sampling** ($T=0.7$) to generate multiple diagnostic leads[cite: 1]. The agent evaluates these leads relative to the execution feedback, effectively "rewarding" strategies that reproduce the bug and "penalizing" those that return empty logs.

### 🗺️ Graph RAG (`ast_graph.py`)
Utilizes a lightweight parser to create a repository skeleton. This provides the agent with a high-level map of classes and functions, preventing "context wandering" during the exploration of large-scale codebases.

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