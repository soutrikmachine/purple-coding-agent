"""
Purple Coding Agent — Multi-Turn Pipeline (v3)
================================================
3-Stage Architecture:

STAGE 1 — LLM LOCALIZATION (Turn 1, via GitHub API)
  - Fetch repo tree
  - LLM reasons: "which files need changing?"
  - Returns bash action to explore those files in the real repo

STAGE 2 — MCTS REPAIR (Final turn)
  - 3 parallel patch branches using all exploration context
  - PRM selects the highest-scoring valid diff
  - Returns patch

Why multi-turn beats single-turn:
  The model sees real stack traces, real function bodies at exact line
  numbers, real test failure messages — not guesses from static GitHub
  snapshots. One "pytest --tb=short" output tells you more than all
  the file fetching we did before.

Green agent behavior (confirmed from logs):
  - Sends: problem_statement, repo, base_commit, instance_id
  - Executes bash actions in the repo's Docker container
  - Returns stdout/stderr in next turn's message
  - Withholds: fail_to_pass, requirements, interface, test_patch
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("purple_agent")

# ── Config ────────────────────────────────────────────────────────────────────

LLM_BASE_URL  = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
MODEL_NAME    = os.getenv("MODEL_NAME", "google/gemma-4-31b-it")
API_KEY       = (
    os.getenv("OPENROUTER_API_KEY", "")
    or os.getenv("LLM_API_KEY", "")
    or os.getenv("HF_TOKEN", "")
)
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")
PORT          = int(os.getenv("PORT", "9010"))
MCTS_BRANCHES = int(os.getenv("MCTS_BRANCHES", "3"))
USE_MCTS      = os.getenv("USE_MCTS", "true").lower() == "true"

CHAT_URL = (
    f"{LLM_BASE_URL}/chat/completions"
    if "/v1" in LLM_BASE_URL
    else f"{LLM_BASE_URL}/v1/chat/completions"
)

logger.info("=" * 60)
logger.info("Purple Agent  model=%s", MODEL_NAME)
logger.info("Chat URL      %s", CHAT_URL)
logger.info("API Key       %s", "SET ✓" if API_KEY else "MISSING ✗")
logger.info("GitHub Token  %s", "SET ✓" if GITHUB_TOKEN else "NOT SET")
logger.info("pipeline      MCTS_BRANCHES=%d", MCTS_BRANCHES)
logger.info("=" * 60)


# ==============================================================================
# GITHUB HELPERS  (Stage 1 only — tree + localization)
# ==============================================================================

def _github_headers() -> dict:
    h = {"Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def fetch_file_raw(repo: str, ref: str, filepath: str) -> str:
    """Fetch a single file from raw.githubusercontent.com."""
    url = f"https://raw.githubusercontent.com/{repo}/{ref or 'HEAD'}/{filepath}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            if r.status == 200:
                return r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        logger.warning("GitHub 404: %s (%d)", filepath, e.code)
    except Exception as e:
        logger.warning("GitHub error %s: %s", filepath, e)
    return ""


def get_repo_tree(repo: str, ref: str) -> list[str]:
    """Get all blob paths via GitHub Tree API (recursive)."""
    url = f"https://api.github.com/repos/{repo}/git/trees/{ref or 'HEAD'}?recursive=1"
    try:
        req = urllib.request.Request(url, headers=_github_headers())
        with urllib.request.urlopen(req, timeout=20) as r:
            data  = json.loads(r.read().decode())
            paths = [i["path"] for i in data.get("tree", []) if i["type"] == "blob"]
            logger.info("Repo tree: %d files in %s", len(paths), repo)
            return paths
    except Exception as e:
        logger.warning("Tree API failed for %s: %s", repo, e)
        return []


_EXCLUDE_DIRS = {
    "vendor", "node_modules", "__pycache__", ".git", "dist", "build",
    "testdata", "fixtures", "migrations", "generated", "proto",
}

_SOURCE_EXT = re.compile(
    r'\.(py|go|js|ts|tsx|jsx|java|rb|rs|c|cpp|h|php|cs|swift|kt|vue|svelte)$'
)


def _filter_tree(paths: list[str], max_paths: int = 300) -> list[str]:
    """Filter to source files, exclude vendor/generated dirs, cap at max_paths."""
    filtered = [
        p for p in paths
        if _SOURCE_EXT.search(p)
        and not any(part in _EXCLUDE_DIRS for part in p.split("/"))
    ]
    filtered.sort(key=lambda p: (len(p.split("/")), p))
    return filtered[:max_paths]


# ==============================================================================
# STAGE 1 — LLM LOCALIZATION
# ==============================================================================

def llm_localize(
    problem_statement: str,
    repo: str,
    tree_paths: list[str],
    llm: "LLMClient",
) -> list[str]:
    """
    Ask the LLM: given this bug report and file tree, which files need changing?
    Returns up to 5 file paths that exist in the tree.
    """
    filtered = _filter_tree(tree_paths, max_paths=300)
    if not filtered:
        return []

    system = (
        "You are a senior software engineer performing fault localization.\n"
        "Given a bug report and a repository file tree, identify which source "
        "files are most likely to contain the bug.\n\n"
        "Return ONLY a JSON array of file paths. Example:\n"
        '["lib/config/cluster.go", "lib/services/auth/login.go"]\n\n'
        "Rules:\n"
        "- Maximum 5 files\n"
        "- Use EXACT paths from the file tree\n"
        "- Prefer source files over test files\n"
        "- Think about which module/package owns the described behavior"
    )

    user = (
        f"Repository: {repo}\n\n"
        f"Bug Report:\n{problem_statement[:3000]}\n\n"
        f"File Tree:\n" + "\n".join(filtered) + "\n\n"
        "Which files need to change? Return a JSON array."
    )

    raw = llm.complete(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
        temperature=0.1,
        max_tokens=256,
    )
    if not raw:
        return []

    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    tree_set = set(tree_paths)
    for attempt in [raw, re.search(r'\[.*?\]', raw, re.DOTALL)]:
        text = attempt if isinstance(attempt, str) else (attempt.group(0) if attempt else "")
        try:
            paths = json.loads(text)
            if isinstance(paths, list):
                valid = [p for p in paths if isinstance(p, str) and p in tree_set]
                logger.info("Localization → %s", valid)
                return valid[:5]
        except (json.JSONDecodeError, ValueError):
            continue
    logger.warning("Localization parse failed: %s", raw[:150])
    return []


# ==============================================================================
# TASK & SESSION MODEL
# ==============================================================================

@dataclass
class SWETask:
    problem_statement: str
    cwd: str = "/workspace/repo"
    hints_text: str = ""
    repo: str = ""
    instance_id: str = ""
    base_commit: str = ""


@dataclass
class Observation:
    """One round of bash output from the green agent."""
    command: str
    stdout: str
    stderr: str

    def render(self) -> str:
        parts = [f"$ {self.command}"]
        if self.stdout.strip():
            out = self.stdout.strip()
            if len(out) > 3000:
                out = out[:1500] + "\n…[truncated]…\n" + out[-500:]
            parts.append(out)
        if self.stderr.strip():
            err = self.stderr.strip()
            if len(err) > 800:
                err = err[:400] + "\n…[truncated]…\n" + err[-200:]
            parts.append(f"[stderr] {err}")
        return "\n".join(parts)


# ==============================================================================
# MCTS ENGINE
# ==============================================================================

EXPLORATION_C = math.sqrt(2)


@dataclass
class MCTSNode:
    state: dict
    parent: "MCTSNode | None" = None
    action: dict | None = None
    children: list["MCTSNode"] = field(default_factory=list)
    _visits: int = 0
    _value_sum: float = 0.0

    @property
    def value(self) -> float:
        return self._value_sum / self._visits if self._visits else 0.0

    def uct(self, parent_visits: int) -> float:
        if self._visits == 0:
            return float("inf")
        return self.value + EXPLORATION_C * math.sqrt(
            math.log(parent_visits + 1) / self._visits
        )

    def update(self, reward: float):
        self._visits += 1
        self._value_sum += reward

    def best_child(self) -> "MCTSNode | None":
        return max(self.children, key=lambda c: c.uct(self._visits)) \
               if self.children else None


class MCTSEngine:
    def __init__(self, branches: int = 3):
        self.root     = MCTSNode(state={})
        self.branches = branches
        self._current = self.root
        self._pending: MCTSNode | None = None

    def select(self, candidates: list[tuple[str, float]]) -> str:
        """Add candidate patches as children and return the UCT-best one."""
        for patch, score in candidates:
            child = MCTSNode(state={}, parent=self._current,
                             action={"patch": patch})
            child.update(score)
            self._current.children.append(child)
        best = self._current.best_child()
        if best is None:
            return candidates[0][0]
        self._pending = best
        return best.action["patch"]

    def backpropagate(self, reward: float):
        node = self._pending or self._current
        while node:
            node.update(reward)
            node = node.parent
        if self._pending:
            self._current = self._pending
            self._pending = None


# ==============================================================================
# PRM (PROCESS REWARD MODEL)
# ==============================================================================

class ProgrammablePRM:
    """Score a patch candidate without executing it."""

    def score(self, patch: str, task: SWETask) -> float:
        if not patch.strip():
            return 0.0
        return (
            0.35 * self._format(patch)
            + 0.35 * self._relevance(patch, task)
            + 0.30 * self._completeness(patch)
        )

    def _format(self, patch: str) -> float:
        return min(
            0.3 * ("diff --git" in patch)
            + 0.2 * ("@@" in patch)
            + 0.2 * bool(re.search(r'^-[^-]', patch, re.MULTILINE))
            + 0.2 * bool(re.search(r'^\+[^\+]', patch, re.MULTILINE))
            + 0.1 * ("\n" in patch),
            1.0,
        )

    def _relevance(self, patch: str, task: SWETask) -> float:
        ps_tok = set(re.findall(r'\b\w{4,}\b', task.problem_statement.lower()))
        ct_tok = set(re.findall(r'\b\w{4,}\b', patch.lower()))
        if not ps_tok:
            return 0.5
        return min(len(ps_tok & ct_tok) / max(len(ps_tok) * 0.4, 1), 1.0)

    def _completeness(self, patch: str) -> float:
        lines  = patch.strip().splitlines()
        n_plus = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
        return 0.0 if n_plus == 0 else (0.2 if len(patch) < 50 else 1.0)


# ==============================================================================
# LLM CLIENT
# ==============================================================================

class LLMClient:
    def __init__(self):
        self._headers = {"Content-Type": "application/json"}
        if API_KEY:
            self._headers["Authorization"] = f"Bearer {API_KEY}"
            logger.info("LLM auth token set ✓")
        else:
            logger.warning("LLM: no API token — calls will fail")

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> str:
        payload: dict[str, Any] = {
            "model":       MODEL_NAME,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            # NOTE: Do NOT add "reasoning" parameter — causes content=null in Gemma 4
        }
        for attempt in range(1, 4):
            try:
                resp = requests.post(
                    CHAT_URL, json=payload,
                    headers=self._headers, timeout=90,
                )
                logger.info("LLM status: %d", resp.status_code)
                if resp.status_code == 401:
                    logger.error("LLM 401 — check OPENROUTER_API_KEY secret")
                    return ""
                if resp.status_code == 429:
                    time.sleep(5 * attempt)
                    continue
                if resp.status_code != 200:
                    logger.error("LLM %d: %s", resp.status_code, resp.text[:300])
                    return ""
                content = resp.json()["choices"][0]["message"]["content"]
                if content is None:
                    logger.error("LLM returned content=null — check model config")
                    return ""
                logger.info("LLM returned %d chars", len(content))
                return content
            except requests.RequestException as e:
                logger.warning("LLM attempt %d/3: %s", attempt, e)
                if attempt < 3:
                    time.sleep(2 ** attempt)
                else:
                    return ""
        return ""


# ==============================================================================
# PROMPT BUILDERS
# ==============================================================================

# System prompt shared across all bash exploration turns
_EXPLORE_SYSTEM = """\
You are an expert software engineer debugging a real GitHub issue in a live repository.
You have an interactive bash shell in the project root.

Work in two phases:
  EXPLORE — run bash commands to understand the bug
  PATCH   — when ready, output the final unified diff

ALWAYS respond in this exact XML format:
<thought>
Your step-by-step reasoning. What did you learn? What do you need next?
</thought>
<action>bash|patch</action>
<content>
bash command OR unified diff here
</content>

EXPLORATION STRATEGY:
1. Use grep to find the exact functions/methods mentioned in the bug report
2. Use cat -n to read files with real line numbers
3. Run the relevant tests with: python -m pytest <path> -x --tb=short 2>&1 | head -50
   or for Go: go test ./... -run <TestName> 2>&1 | head -50
4. Trace the actual error to the exact line before patching

BASH RULES:
- Commands must be single-line or use && to chain
- Prefer: grep -n, cat -n, find, head, tail
- No destructive commands (rm, dd, git push)
- Keep output manageable with | head -N

PATCH RULES (when action=patch):
- Start with: diff --git a/file b/file
- Copy context lines CHARACTER-FOR-CHARACTER from what cat -n showed you
- Include exactly 3 unchanged context lines before/after each change
- No markdown fences, no explanation — raw diff only
"""

def _build_repair_messages(
    task: SWETask,
    history: list[Observation],
    located_paths: list[str],
    fetched_files: dict[str, str],
) -> list[dict]:
    """
    Build the final patch generation prompt.
    Includes full exploration history so the model has maximum context.
    """
    # Format exploration history as a readable log
    exploration_log = ""
    if history:
        parts = []
        for obs in history:
            parts.append(obs.render())
        exploration_log = (
            "\n\n## Exploration History (bash commands + real output)\n\n"
            + "\n\n".join(parts)
        )

    # Add any statically fetched files as fallback context
    file_context = ""
    if fetched_files:
        sections = []
        for fp, content in fetched_files.items():
            lines = content.splitlines()
            if len(lines) > 120:
                content = "\n".join(lines[:120]) + f"\n[...{len(lines)-120} more lines]"
            sections.append(f"### {fp}\n```\n{content}\n```")
        file_context = "\n\n## Static File Context\n\n" + "\n\n".join(sections)

    located_hint = ""
    if located_paths:
        located_hint = (
            "\n\nFiles identified for modification:\n"
            + "\n".join(f"  - {p}" for p in located_paths)
        )

    system = (
        "You are an expert software engineer. Based on your exploration, "
        "generate the exact patch to fix the described bug.\n\n"
        "OUTPUT FORMAT: unified diff ONLY, starting with diff --git\n"
        "- Copy context lines CHARACTER-FOR-CHARACTER from what you saw\n"
        "- Include exactly 3 unchanged context lines before/after changes\n"
        "- May span multiple files\n"
        "- No <think> tags, no markdown fences, no explanation\n"
    )

    user = (
        f"Repository: {task.repo}  commit: {task.base_commit or 'HEAD'}\n\n"
        f"## Bug Report\n\n{task.problem_statement[:2000]}"
        f"{located_hint}"
        f"{exploration_log}"
        f"{file_context}\n\n"
        "Now output the final unified diff patch:"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ==============================================================================
# PATCH EXTRACTION
# ==============================================================================

def _extract_patch(raw: str) -> str:
    """Extract a clean unified diff from LLM output."""
    if not raw:
        return ""
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    if "```" in raw:
        m = re.search(r"```(?:diff|patch)?\n(.*?)```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
    if not (raw.startswith("diff --git") or raw.startswith("--- ")):
        m = re.search(r"(diff --git.*)", raw, re.DOTALL)
        raw = m.group(1).strip() if m else ""
    return raw


def _parse_action(raw: str) -> tuple[str, str]:
    """Parse <action> and <content> tags from LLM response."""
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    def tag(t: str) -> str:
        m = re.search(rf'<{t}>(.*?)</{t}>', raw, re.DOTALL)
        return m.group(1).strip() if m else ""

    action  = tag("action").lower() or "bash"
    content = tag("content") or ""

    if action not in ("bash", "patch"):
        # Heuristic fallback
        action = "patch" if ("diff --git" in content or "--- a/" in content) else "bash"

    return action, content


# ==============================================================================
# PURPLE AGENT
# ==============================================================================

class PurpleAgent:
    """
    Multi-turn agent with 3-stage pipeline:
      Stage 1: LLM localization (GitHub tree → LLM → file paths)
      Stage 2: Bash exploration (interactive shell in repo Docker container)
      Stage 3: MCTS repair (parallel patch generation with PRM selection)
    """

    def __init__(self):
        self.llm      = LLMClient()
        self.prm      = ProgrammablePRM()
        self._sessions: dict[str, dict[str, Any]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def respond(self, message: dict) -> dict:
        """Entry point called from FastAPI handler (runs in thread pool)."""
        session_id = (
            message.get("session_id")
            or message.get("instance_id")
            or str(abs(hash(message.get("problem_statement", "")[:80])))
        )

        if session_id not in self._sessions:
            session = self._init_session(session_id, message)
        else:
            session = self._sessions[session_id]
            # Record observation from previous bash action
            self._record_observation(session, message)

        try:
            return asyncio.run(self._step(session))
        except Exception as e:
            logger.exception("[%s] Agent crashed: %s", session_id[:20], e)
            return {"action": "patch", "content": ""}

    # ── Session Management ────────────────────────────────────────────────────

    def _init_session(self, session_id: str, message: dict) -> dict:
        task = SWETask(
            problem_statement=message.get("problem_statement", ""),
            cwd=message.get("cwd", "/workspace/repo"),
            hints_text=message.get("hints_text", ""),
            repo=message.get("repo", ""),
            instance_id=message.get("instance_id", ""),
            base_commit=message.get("base_commit", ""),
        )
        session = {
            "id":             session_id,
            "task":           task,
            "history":        [],           # list[Observation]
            "last_command":   "",           # last bash command sent
            "located_paths":  None,         # set after Stage 1
            "fetched_files":  {},           # static files from GitHub
            "mcts":           MCTSEngine(branches=MCTS_BRANCHES),
        }
        self._sessions[session_id] = session
        logger.info("[%s] New session  repo=%s  commit=%s",
                    session_id[:20], task.repo,
                    task.base_commit[:12] if task.base_commit else "HEAD")
        return session

    def _record_observation(self, session: dict, message: dict):
        """Store stdout/stderr returned by the green agent into session history."""
        stdout = message.get("stdout", "").strip()
        stderr = message.get("stderr", "").strip()
        if stdout or stderr:
            obs = Observation(
                command=session.get("last_command", ""),
                stdout=stdout,
                stderr=stderr,
            )
            session["history"].append(obs)
            logger.info("[%s] Observation recorded: stdout=%d chars stderr=%d chars",
                        session["id"][:20], len(stdout), len(stderr))

    # ── Main Step Logic ───────────────────────────────────────────────────────

    async def _step(self, session: dict) -> dict:
        """Single-turn: localize → fetch → MCTS repair → return patch."""
        task  = session["task"]

        logger.info("[%s] Running pipeline", session["id"][:20])

        # ── Stage 1: Localization on first turn ───────────────────────────────
        if session["located_paths"] is None:
            await self._run_localization(session)

        # Stage 2: MCTS repair — always return patch, green agent is single-turn
        # (bash exploration requires Docker socket which we don't have)
        logger.info("[%s] Running MCTS repair", session["id"][:20])
        patch = await self._mcts_repair(session)
        return {"action": "patch", "content": patch}

    # ── Stage 1: LLM Localization ─────────────────────────────────────────────

    async def _run_localization(self, session: dict):
        """Fetch repo tree, ask LLM which files need changing, fetch them."""
        task = session["task"]

        tree = await asyncio.to_thread(
            get_repo_tree, task.repo, task.base_commit or "HEAD"
        )

        if tree:
            located = await asyncio.to_thread(
                llm_localize,
                task.problem_statement,
                task.repo,
                tree,
                self.llm,
            )
            session["located_paths"] = located

            # Fetch located files from GitHub as static context fallback
            ref = task.base_commit or "HEAD"
            for fp in located[:4]:
                content = await asyncio.to_thread(
                    fetch_file_raw, task.repo, ref, fp
                )
                if content:
                    session["fetched_files"][fp] = content
                    logger.info("Fetched: %s (%d chars)", fp, len(content))
        else:
            session["located_paths"] = []
            logger.warning("[%s] Tree fetch failed", session["id"][:20])


    # ── Stage 3: MCTS Repair ──────────────────────────────────────────────────

    async def _mcts_repair(self, session: dict) -> str:
        """
        Generate MCTS_BRANCHES patch candidates in parallel.
        Temperature schedule spans 0.15 → 0.80 for diversity.
        PRM selects the highest-scoring valid patch.
        """
        task = session["task"]
        msgs = _build_repair_messages(
            task,
            session["history"],
            session["located_paths"] or [],
            session["fetched_files"],
        )

        temps = [
            0.15 + (0.65 / max(MCTS_BRANCHES - 1, 1)) * i
            for i in range(MCTS_BRANCHES)
        ]

        # Fire all branches simultaneously
        raws = await asyncio.gather(*[
            asyncio.to_thread(self.llm.complete, msgs, t, 2048)
            for t in temps
        ])

        candidates: list[tuple[str, float]] = []
        for i, raw in enumerate(raws):
            patch = _extract_patch(raw)
            score = self.prm.score(patch, task)
            candidates.append((patch, score))
            logger.info("[%s] Branch %d/%d T=%.2f score=%.3f len=%d",
                        session["id"][:20], i+1, MCTS_BRANCHES,
                        temps[i], score, len(patch))

        if not candidates or all(s == 0 for _, s in candidates):
            logger.warning("[%s] All branches scored 0 — returning best effort",
                           session["id"][:20])
            # Return the longest non-empty patch as last resort
            return max((p for p, _ in candidates if p), key=len, default="")

        best_patch = session["mcts"].select(candidates)
        best_score = next(s for p, s in candidates if p == best_patch)
        session["mcts"].backpropagate(best_score)

        logger.info("[%s] MCTS selected score=%.3f len=%d",
                    session["id"][:20], best_score, len(best_patch))
        return best_patch


# ==============================================================================
# FASTAPI APP
# ==============================================================================

app   = FastAPI(title="Purple Coding Agent")
agent = PurpleAgent()

AGENT_CARD = {
    "name": "Purple Coding Agent",
    "description": (
        "Multi-turn SWE-bench agent: LLM localization → bash exploration → "
        "MCTS repair. Gemma 4 31B + GitHub API."
    ),
    "url": f"http://localhost:{PORT}/",
    "version": "3.0.0",
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": False,
    },
    "defaultInputModes": ["application/json"],
    "defaultOutputModes": ["application/json"],
    "skills": [
        {
            "id": "swe_patch",
            "name": "SWE Patch",
            "description": (
                "Interactively explore a repository via bash, then generate "
                "a unified diff patch fixing the described issue."
            ),
            "tags": ["coding", "swe-bench", "patch", "multi-turn"],
            "examples": [],
        }
    ],
}


@app.get("/.well-known/agent-card.json")
async def agent_card():
    return JSONResponse(content=AGENT_CARD)


@app.get("/.well-known/agent.json")
async def agent_card_compat():
    return JSONResponse(content=AGENT_CARD)


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "purple-coding-agent", "version": "3.0.0"}


@app.post("/")
async def handle_task(request: Request):
    body = await request.json()

    jsonrpc_id  = body.get("id", str(uuid.uuid4()))
    task_id     = str(uuid.uuid4())
    artifact_id = str(uuid.uuid4())

    logger.info("─" * 50)
    logger.info("Request  id=%s  method=%s", jsonrpc_id, body.get("method"))

    task_data, context_id = _extract_task_and_context(body)
    if not context_id:
        context_id = str(uuid.uuid4())

    # Thread context_id as session_id so it persists across ALL turns
    task_data["session_id"] = context_id

    ps = task_data.get("problem_statement", "")
    logger.info("context_id=%s  ps_len=%d  repo=%s  commit=%s",
                context_id[:20], len(ps),
                task_data.get("repo", "?"),
                (task_data.get("base_commit", "") or "")[:12] or "HEAD")

    # Run agent in thread so asyncio.run() inside works correctly
    loop   = asyncio.get_event_loop()
    action = await loop.run_in_executor(None, agent.respond, task_data)

    # Return bash action as JSON, patch as raw diff
    if action.get("action") == "patch":
        artifact_text = action.get("content", "")
    else:
        # bash action — green agent reads this as JSON
        artifact_text = json.dumps(action)

    logger.info("Response: action=%s  len=%d",
                action.get("action"), len(artifact_text))

    return JSONResponse(content={
        "jsonrpc": "2.0",
        "id":      jsonrpc_id,
        "result": {
            "id":        task_id,
            "contextId": context_id,
            "status":    {"state": "completed"},
            "artifacts": [
                {
                    "artifactId": artifact_id,
                    "name":       "patch",
                    "parts":      [{"kind": "text", "text": artifact_text}],
                }
            ],
        },
    })


# ==============================================================================
# MESSAGE EXTRACTION
# ==============================================================================

def _extract_task_and_context(body: dict) -> tuple[dict, str]:
    """
    Parse A2A JSON-RPC envelope from green agent.

    Turn 1 (task delivery):
      params.message.parts[0].text = JSON with problem_statement, repo, etc.

    Turn 2+ (observation):
      params.message.parts[0].text = JSON with stdout, stderr, cwd
      contextId is the SAME as turn 1 — used as session key
    """
    context_id = ""

    if "problem_statement" in body:
        return body, context_id

    try:
        params     = body.get("params", {})
        message    = params.get("message", {})
        context_id = message.get("contextId", "") or params.get("contextId", "")

        parts = message.get("parts", [])
        logger.info("Parts: %d  contextId=%s",
                    len(parts), context_id[:20] if context_id else "none")

        for i, part in enumerate(parts):
            kind = part.get("kind") or part.get("type", "")
            text = part.get("text", "")
            logger.info("Part[%d] kind=%s len=%d preview=%s",
                        i, kind, len(text), text[:150])

            if kind == "data":
                data = part.get("data", {})
                if isinstance(data, dict) and "problem_statement" in data:
                    return data, context_id

            if kind == "text" and text.strip():
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        if "problem_statement" in parsed:
                            return parsed, context_id
                        # Observation turn: stdout/stderr/cwd
                        if any(k in parsed for k in
                               ("stdout", "stderr", "cwd", "repo", "instance_id")):
                            return parsed, context_id
                except (json.JSONDecodeError, ValueError):
                    pass
                return {"problem_statement": text.strip()}, context_id

    except Exception as e:
        logger.error("Extraction error: %s", e)

    ps = _deep_find(body, "problem_statement")
    if ps:
        return {"problem_statement": ps}, context_id

    logger.warning("Nothing found. Body keys: %s", list(body.keys()))
    return {}, context_id


def _deep_find(obj: Any, key: str, depth: int = 0) -> str:
    if depth > 6:
        return ""
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], str):
            return obj[key]
        for v in obj.values():
            r = _deep_find(v, key, depth + 1)
            if r:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = _deep_find(item, key, depth + 1)
            if r:
                return r
    return ""


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, log_level="info")