"""
Purple Coding Agent — Complete Server
======================================
Single-file implementation combining:
  - A2A server (FastAPI, correct JSON-RPC + response format)
  - MCTS engine (UCT-based action selection)
  - Programmable PRM (3-layer reward scoring)
  - Node State Manager (per-branch working set / discovery log)
  - LLM Client (HuggingFace Router / OpenRouter / vLLM)
  - Multi-turn session management

A2A Protocol fixes applied:
  1. Problem extracted from params.message.parts[].text (JSON-RPC format)
  2. Response includes jsonrpc, contextId, artifactId, kind:"text"
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import uuid
import urllib.request
import urllib.error
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
MODEL_NAME    = os.getenv("MODEL_NAME", "deepseek/deepseek-v3.2")
API_KEY       = (
    os.getenv("OPENROUTER_API_KEY", "")
    or os.getenv("LLM_API_KEY", "")
    or os.getenv("HF_TOKEN", "")
)
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")
PORT          = int(os.getenv("PORT", "9010"))
MAX_TURNS     = int(os.getenv("MAX_TURNS", "15"))
MCTS_BRANCHES = int(os.getenv("MCTS_BRANCHES", "3"))
TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.6"))
USE_MCTS      = os.getenv("USE_MCTS", "true").lower() == "true"

# Correct chat completions URL (no double /v1)
CHAT_URL = (
    f"{LLM_BASE_URL}/chat/completions"
    if "/v1" in LLM_BASE_URL
    else f"{LLM_BASE_URL}/v1/chat/completions"
)

logger.info("=" * 60)
logger.info("Purple Agent  model=%s", MODEL_NAME)
logger.info("Chat URL      %s", CHAT_URL)
logger.info("API Key       %s", "SET ✓" if API_KEY else "MISSING ✗")
logger.info("GitHub Token  %s", "SET ✓" if GITHUB_TOKEN else "NOT SET (public repos only)")
logger.info("MCTS          branches=%d  max_turns=%d  enabled=%s",
            MCTS_BRANCHES, MAX_TURNS, USE_MCTS)
logger.info("=" * 60)

#==============================================================================
# GitHub file fetching 
#==============================================================================

def _github_headers() -> dict:
    """Build headers for GitHub API calls."""
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h
 
 
def fetch_file_raw(repo: str, ref: str, filepath: str) -> str:
    """
    FIX 1: Fetch raw file content using raw.githubusercontent.com.
    This is simpler and more reliable than the Contents API.
    Works for public repos without any auth token.
 
    URL: https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{filepath}
    """
    # ref can be a commit SHA or branch name; use HEAD if empty
    ref = ref or "HEAD"
    url = f"https://raw.githubusercontent.com/{repo}/{ref}/{filepath}"
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            if r.status == 200:
                content = r.read().decode("utf-8", errors="replace")
                logger.info("GitHub raw fetch OK: %s (%d chars)", filepath, len(content))
                return content
    except urllib.error.HTTPError as e:
        logger.warning("GitHub fetch failed for %s: HTTP %d", filepath, e.code)
    except Exception as e:
        logger.warning("GitHub fetch failed for %s: %s", filepath, e)
    return ""
 
 
def search_github_for_file(repo: str, filename: str) -> list[str]:
    """
    FIX 2: Use GitHub Code Search API to find the actual path of a file
    within the repository. Essential for monorepos where paths are deep.
 
    Returns a list of file paths found (e.g. ['packages/foo/bar/TotpInput.tsx'])
    """
    # Strip extension for broader search
    name_no_ext = re.sub(r'\.\w+$', '', filename)
    query = f"{name_no_ext} repo:{repo}"
    url   = f"https://api.github.com/search/code?q={urllib.parse.quote(query)}&per_page=5"
 
    try:
        import urllib.parse
        req = urllib.request.Request(url, headers=_github_headers())
        with urllib.request.urlopen(req, timeout=10) as r:
            data  = json.loads(r.read().decode())
            items = data.get("items", [])
            paths = [item["path"] for item in items]
            logger.info("GitHub search '%s' → %s", filename, paths[:3])
            return paths
    except Exception as e:
        logger.warning("GitHub search failed for '%s': %s", filename, e)
        return []
 
 
def fetch_relevant_files(task: "SWETask") -> dict[str, str]:
    """
    FIX 3: Improved file fetching strategy:
    1. Extract candidate filenames from problem statement and test paths
    2. Try direct fetch first (if full path known)
    3. Fall back to GitHub Search API to find actual path in repo
    Returns {filepath: content} for up to 4 files.
    """
    if not task.repo:
        return {}
 
    ref = task.base_commit or "HEAD"
 
    # ── Collect candidate paths ───────────────────────────────────────────────
 
    # From test paths (most reliable — these ARE real paths)
    test_paths: list[str] = []
    for test in task.fail_to_pass[:4]:
        test_file = test.split("::")[0]
        test_paths.append(test_file)
 
    # From problem statement: full paths (e.g. `lib/foo/bar.py`)
    ps_full_paths = re.findall(
        r'(?:^|[\s`"\'])('
        r'[\w][\w/.-]+\.(?:py|go|js|ts|tsx|jsx|java|rb|rs|c|cpp|h|php|cs|swift|kt)'
        r')',
        task.problem_statement,
        re.MULTILINE,
    )
    ps_full_paths = [p.strip() for p in ps_full_paths][:5]
 
    # From problem statement: bare filenames in backticks
    ps_filenames = re.findall(
        r'`([\w.-]+\.(?:py|go|js|ts|tsx|jsx|java|rb|rs|c|cpp|h|php|cs|swift|kt))`',
        task.problem_statement,
    )[:5]
 
    # ── Attempt fetches ───────────────────────────────────────────────────────
 
    files: dict[str, str] = {}
 
    # Try test paths first — they are real paths
    for fp in test_paths:
        if fp and len(files) < 4:
            content = fetch_file_raw(task.repo, ref, fp)
            if content:
                files[fp] = content
 
    # Try full paths from problem statement
    for fp in ps_full_paths:
        if fp and len(files) < 4:
            content = fetch_file_raw(task.repo, ref, fp)
            if content:
                files[fp] = content
 
    # For bare filenames: search the repo to find actual path
    import urllib.parse
    for filename in ps_filenames:
        if len(files) >= 4:
            break
        if any(filename in fp for fp in files):
            continue  # already fetched
        found_paths = search_github_for_file(task.repo, filename)
        for fp in found_paths[:2]:
            if len(files) >= 4:
                break
            content = fetch_file_raw(task.repo, ref, fp)
            if content:
                files[fp] = content
 
    logger.info("Fetched %d files from GitHub for %s", len(files), task.repo)
    return files

# ==============================================================================
# MCTS ENGINE
# ==============================================================================

EXPLORATION_C = math.sqrt(2)


@dataclass
class MCTSNode:
    state: dict
    parent: MCTSNode | None = None
    action: dict | None = None
    children: list[MCTSNode] = field(default_factory=list)
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

    def best_child(self) -> MCTSNode | None:
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.uct(self._visits))

    def is_leaf(self) -> bool:
        return not self.children


class MCTSEngine:
    def __init__(self, root: MCTSNode, branches: int = 3):
        self.root = root
        self.branches = branches
        self._current = root
        self._pending: MCTSNode | None = None

    def select_action(
        self, candidates: list[tuple[dict, float]]
    ) -> tuple[dict, MCTSNode]:
        for action, score in candidates:
            child = MCTSNode(
                state=self._current.state.copy(),
                parent=self._current,
                action=action,
            )
            child.update(score)
            self._current.children.append(child)

        best = self._current.best_child()
        if best is None:
            return candidates[0][0], self._current

        self._pending = best
        logger.debug("MCTS select action=%s uct=%.3f", best.action.get("action"), best.uct(self._current._visits))
        return best.action, best

    def backpropagate(self, reward: float):
        node = self._pending or self._current
        while node is not None:
            node.update(reward)
            node = node.parent
        if self._pending:
            self._current = self._pending
            self._pending = None

    def stats(self) -> dict:
        return {
            "nodes": self._count(self.root),
            "depth": self._depth(self._current),
            "root_value": round(self.root.value, 3),
        }

    @staticmethod
    def _count(n: MCTSNode) -> int:
        return 1 + sum(MCTSEngine._count(c) for c in n.children)

    @staticmethod
    def _depth(n: MCTSNode) -> int:
        d = 0
        while n.parent:
            d += 1
            n = n.parent
        return d


# ==============================================================================
# PROGRAMMABLE PRM (Process Reward Model)
# ==============================================================================

class ProgrammablePRM:
    """
    Three-layer reward scoring:
      Hard  (20%) — format validity
      Soft  (35%) — relevance to problem
      Exec  (45%) — execution signals from observation
    """

    def score_static(self, action: dict, task: SWETask) -> float:
        a = action.get("action", "bash")
        c = action.get("content", "")
        score = 0.20 * self._format(a, c) + 0.35 * self._relevance(c, task)
        return min(score, 1.0)

    def score_observation(self, obs: dict, task: SWETask) -> float:
        stdout = obs.get("stdout", "")
        stderr = obs.get("stderr", "")
        score  = 0.3  # baseline
        score += 0.45 * self._exec(stdout, stderr, task)
        score += 0.15 * self._discovery(stdout, task)
        if stderr and not stdout:
            score -= 0.10
        return max(0.0, min(score, 1.0))

    def _format(self, action_type: str, content: str) -> float:
        if not content.strip():
            return 0.0
        if action_type == "patch":
            has_header = "diff --git" in content or "--- a/" in content
            has_hunk   = "@@" in content
            has_change = bool(re.search(r'^\+', content, re.MULTILINE))
            return (0.4 * has_header) + (0.3 * has_hunk) + (0.3 * has_change)
        return 1.0 if len(content) < 500 else 0.5

    def _relevance(self, content: str, task: SWETask) -> float:
        if not task.problem_statement:
            return 0.5
        ps_tok = set(re.findall(r"\b\w{4,}\b", task.problem_statement.lower()))
        ct_tok = set(re.findall(r"\b\w{4,}\b", content.lower()))
        if not ps_tok:
            return 0.5
        return min(len(ps_tok & ct_tok) / len(ps_tok) * 2, 1.0)

    def _exec(self, stdout: str, stderr: str, task: SWETask) -> float:
        sl = stdout.lower()
        if re.search(r"\d+ passed", sl) and "failed" not in sl:
            return 1.0
        if "passed" in sl and "failed" not in sl:
            return 0.9
        if not stderr and stdout.strip():
            return 0.6
        if "failed" in sl or "error" in stderr.lower():
            return 0.2
        return 0.4

    def _discovery(self, stdout: str, task: SWETask) -> float:
        files = re.findall(r'[\w/.-]+\.py(?::\d+)?', stdout)
        if not files:
            return 0.0
        ps_lower = task.problem_statement.lower()
        for fp in files:
            base = fp.split("/")[-1].replace(".py", "")
            if base in ps_lower:
                return 1.0
        return 0.5


# ==============================================================================
# NODE STATE MANAGER
# ==============================================================================

@dataclass
class NodeState:
    cwd: str = "/workspace/repo"
    working_set: list[str] = field(default_factory=list)
    discovery_log: dict[str, str] = field(default_factory=dict)
    current_patch: str = ""

    def copy(self) -> NodeState:
        return NodeState(
            cwd=self.cwd,
            working_set=list(self.working_set),
            discovery_log=dict(self.discovery_log),
            current_patch=self.current_patch,
        )

    def add_file(self, path: str):
        if path not in self.working_set:
            self.working_set.append(path)

    def summarize(self) -> str:
        parts = []
        if self.working_set:
            parts.append("Open files: " + ", ".join(self.working_set[-5:]))
        if self.discovery_log:
            parts.append("Discoveries:")
            for loc, fact in list(self.discovery_log.items())[-6:]:
                parts.append(f"  • {loc}: {fact}")
        if self.current_patch:
            parts.append(f"Current patch: {len(self.current_patch.splitlines())} lines")
        return "\n".join(parts) or "No state yet."


# ==============================================================================
# TASK MODEL
# ==============================================================================

@dataclass
class SWETask:
    problem_statement: str
    cwd: str = "/workspace/repo"
    hints_text: str = ""
    python_version: str = "3.9"
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    repo: str = ""
    instance_id: str = ""
    base_commit: str = ""


# ==============================================================================
# PROMPTS
# ==============================================================================

SYSTEM_PROMPT = """\
You are a world-class software engineer debugging a real GitHub issue.

You have access to an interactive shell in the project repository.
You work in three modes:

  bash   — Read-only exploration (find, grep, cat, git log, python -c, pytest)
  debug  — Temporary writes + execution to verify hypotheses
  patch  — Submit the final unified diff that fixes the bug

ALWAYS respond in this XML format:

<thought>
Step-by-step analysis. Reference specific file names, line numbers, functions.
Verify your reasoning before acting.
</thought>
<command>bash|debug|patch</command>
<content>
the shell command OR unified diff here
</content>

RULES:
1. Start with: find /workspace/repo -type f -name "*.py" | head -30
2. Read relevant files with cat -n to see line numbers
3. Run failing tests: python -m pytest <test_path> -x --tb=short
4. Only switch to patch mode when confident
5. Patch must be valid unified diff (git diff format), starting with diff --git
6. Never use rm, dd, or destructive commands
"""


def build_user_prompt(task: SWETask) -> str:
    lines = ["# Bug Report", "", task.problem_statement.strip()]
    if task.hints_text:
        lines += ["", "## Hints", task.hints_text.strip()]
    if task.fail_to_pass:
        lines += ["", "## Tests That Must Pass After Fix"]
        lines += [f"  - {t}" for t in task.fail_to_pass]
    lines += [
        "", "## Environment",
        f"  Working directory : {task.cwd}",
        f"  Python version    : {task.python_version}",
        "", "Begin your exploration. Output your first action.",
    ]
    return "\n".join(lines)


def build_obs_prompt(obs: dict) -> str:
    parts = []
    cwd = obs.get("cwd", "")
    if cwd:
        parts.append(f"[cwd: {cwd}]")
    stdout = obs.get("stdout", "").strip()
    stderr = obs.get("stderr", "").strip()
    if stdout:
        if len(stdout) > 3000:
            stdout = stdout[:1500] + "\n…[truncated]…\n" + stdout[-400:]
        parts.append(f"stdout:\n```\n{stdout}\n```")
    if stderr:
        if len(stderr) > 800:
            stderr = stderr[:400] + "\n…[truncated]…\n" + stderr[-200:]
        parts.append(f"stderr:\n```\n{stderr}\n```")
    if not stdout and not stderr:
        parts.append("(no output)")
    return "\n".join(parts) + "\n\nWhat is your next action?"


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
            logger.warning("LLM: no API token — calls will 401")

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.6,
        max_tokens: int = 2048,
    ) -> str:
        payload: dict[str, Any] = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        for attempt in range(1, 4):
            try:
                resp = requests.post(
                    CHAT_URL,
                    json=payload,
                    headers=self._headers,
                    timeout=180,
                )
                if resp.status_code == 401:
                    logger.error("LLM 401 Unauthorized — check OPENROUTER_API_KEY secret")
                    return ""
                if resp.status_code == 429:
                    wait = 5 * attempt
                    logger.warning("LLM 429 rate limited — waiting %ds", wait)
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    logger.error("LLM %d: %s", resp.status_code, resp.text[:200])
                    return ""
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                logger.info("LLM status=200 returned %d chars", len(content))
                return content
            except requests.RequestException as e:
                logger.warning("LLM attempt %d/3: %s", attempt, e)
                if attempt < 3:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("All LLM retries exhausted")
                    return ""
        return ""


# ==============================================================================
# PURPLE AGENT (MCTS ORCHESTRATOR)
# ==============================================================================

class PurpleAgent:
    """
    Stateful agent — one session per instance_id.
    Each call to respond() handles one A2A turn.
    """

    def __init__(self):
        self.llm  = LLMClient()
        self.prm  = ProgrammablePRM()
        self._sessions: dict[str, dict[str, Any]] = {}

    def respond(self, message: dict) -> dict:
        session_id = (
            message.get("session_id")
            or message.get("instance_id")
            or str(abs(hash(message.get("problem_statement", "")[:80])))
        )

        if session_id not in self._sessions:
            session = self._init_session(session_id, message)
        else:
            session = self._sessions[session_id]

        try:
            return self._step(session, message)
        except Exception as e:
            logger.exception("[%s] Agent step crashed: %s", session_id, e)
            return {"action": "patch", "content": ""}

    # ── Session Init ──────────────────────────────────────────────────────────

    def _init_session(self, session_id: str, message: dict) -> dict:
        task = SWETask(
            problem_statement=message.get("problem_statement", ""),
            cwd=message.get("cwd", "/workspace/repo"),
            hints_text=message.get("hints_text", ""),
            python_version=message.get("python_version", "3.9"),
            fail_to_pass=message.get("fail_to_pass", []) or [],
            pass_to_pass=message.get("pass_to_pass", []) or [],
            repo=message.get("repo", ""),
            instance_id=message.get("instance_id", ""),
            base_commit=message.get("base_commit", "")
        )
        root = MCTSNode(state=NodeState(cwd=task.cwd).__dict__)
        session = {
            "id": session_id,
            "task": task,
            "mcts": MCTSEngine(root, branches=MCTS_BRANCHES),
            "node_state": NodeState(cwd=task.cwd),
            "turn": 0,
            "history": [],
            "submitted_patch": None,
            "_fetched_files": None,
        }
        self._sessions[session_id] = session
        logger.info("[%s] New session repo=%s", session_id, task.repo)
        return session

    # ── Step ──────────────────────────────────────────────────────────────────
    def _step(self, session: dict, message: dict) -> dict:
        task: SWETask = session["task"]
        session["turn"] += 1
        turn = session["turn"]
 
        # Record observation from green agent
        if "stdout" in message or "stderr" in message:
            obs = {
                "cwd":    message.get("cwd", task.cwd),
                "stdout": message.get("stdout", ""),
                "stderr": message.get("stderr", ""),
            }
            if session["history"]:
                session["history"][-1]["observation"] = obs
                score = self.prm.score_observation(obs, task)
                session["mcts"].backpropagate(score)
                logger.info("[%s] turn=%d PRM=%.3f", session["id"], turn, score)
 
                # Update node state from observation
                self._update_state(session["node_state"], obs)
 
        # Force patch when nearing turn limit
        if turn >= MAX_TURNS - 1 and not session["submitted_patch"]:
            logger.info("[%s] Forcing patch (turn limit)", session["id"])
            return self._force_patch(session)
 
        # Select action
        if USE_MCTS:
            action = self._mcts_patch(session)
        else:
            action = self._greedy_patch(session)
 
        session["history"].append({"action": action})
        logger.info("[%s] turn=%d action=%s", session["id"], turn, action.get("action"))
        return action
    
    # ── Action Selection ──────────────────────────────────────────────────────

    def _greedy_action(self, session: dict) -> dict:
        msgs = self._build_messages(session)
        raw  = self.llm.complete(msgs, temperature=TEMPERATURE, max_tokens=1024)
        return self._parse_action(raw, session)

    def _mcts_action(self, session: dict) -> dict:
        msgs = self._build_messages(session)
        candidates = []
        for _ in range(MCTS_BRANCHES):
            raw    = self.llm.complete(msgs, temperature=TEMPERATURE + 0.1, max_tokens=1024)
            action = self._parse_action(raw, session)
            score  = self.prm.score_static(action, session["task"])
            candidates.append((action, score))
            logger.info("[%s] MCTS branch %d/%d action=%s score=%.3f", session["id"][:20], _ + 1, MCTS_BRANCHES, action.get("action"), score)

        best, _ = session["mcts"].select_action(candidates)
        logger.info("MCTS stats: %s", session["mcts"].stats())
        return best

    def _force_patch(self, session: dict) -> dict:
        task    = session["task"]
        history = self._format_history(session["history"])
        msgs    = [
            {"role": "system", "content": "You are an expert software engineer. Output ONLY a unified diff patch starting with diff --git. No explanation."},
            {"role": "user",   "content": f"Based on exploration:\n{history}\n\nFix:\n{task.problem_statement[:600]}"},
        ]
        raw = self.llm.complete(msgs, temperature=0.15, max_tokens=4096)
        action = self._parse_action(raw, session)
        if action.get("action") != "patch":
            content = action.get("content", "")
            action  = {
                "action":  "patch",
                "content": content if ("diff --git" in content or "--- a/" in content) else "",
            }
        session["submitted_patch"] = action["content"]
        return action

    def _greedy_patch(self, session: dict) -> dict:
        """Single-shot patch generation."""
        msgs = self._build_patch_messages(session)
        raw  = self.llm.complete(msgs, temperature=0.2, max_tokens=4096)
        return self._force_to_patch(raw, session)

    def _mcts_patch(self, session: dict) -> dict:
        """
        MCTS over patch candidates — sample MCTS_BRANCHES patches,
        score each with static PRM, return highest scoring one.
        This is inference-time scaling: more branches = better patch selection.
        """
        msgs = self._build_patch_messages(session)
        candidates = []
        for i in range(MCTS_BRANCHES):
            raw    = self.llm.complete(msgs, temperature=0.3 + i * 0.15, max_tokens=4096)
            action = self._force_to_patch(raw, session)
            score  = self.prm.score_static(action, session["task"])
            candidates.append((action, score))
            logger.info("[%s] MCTS branch %d/%d score=%.3f patch_len=%d",
                       session["id"], i+1, MCTS_BRANCHES, score, len(action.get("content","")))

        # UCT select best
        best_action, best_score = max(candidates, key=lambda x: x[1])
        logger.info("[%s] MCTS selected branch score=%.3f", session["id"], best_score)
        session["mcts"].backpropagate(best_score)
        return best_action

    def _build_patch_messages(self, session: dict) -> list[dict]:
        task = session["task"]
 
        # Fetch files once per session (cached), not once per MCTS branch
        if session["_fetched_files"] is None:
            session["_fetched_files"] = fetch_relevant_files(task)
        real_files = session["_fetched_files"]
 
        # FIX 4: Build system prompt WITHOUT leading indentation
        # The old version used f"""...""" with indented lines — this added
        # spaces to every line which confused the model.
        if real_files:
            file_sections = []
            for filepath, content in real_files.items():
                lines = content.splitlines()
                if len(lines) > 200:
                    # Keep first 100 + last 50 lines for large files
                    content = (
                        "\n".join(lines[:100])
                        + f"\n\n... [{len(lines) - 150} lines omitted] ...\n\n"
                        + "\n".join(lines[-50:])
                    )
                file_sections.append(f"### {filepath}\n```\n{content}\n```")
            file_context = "\n\n## ACTUAL FILE CONTENTS\nUse EXACTLY these lines for your diff:\n\n" + "\n\n".join(file_sections)
            logger.info("[%s] Providing %d real files to model", session["id"][:20], len(real_files))
        else:
            file_context = "\n\n## NOTE\nCould not fetch file content from GitHub. Infer from the issue description and test file paths."
            logger.warning("[%s] No files fetched — model working blind", session["id"][:20])
 
        # Build test hints
        test_hint = ""
        if task.fail_to_pass:
            test_hint = "\n\nFailing tests (these must pass after your fix):\n" + "\n".join(
                f"  - {t}" for t in task.fail_to_pass[:5]
            )
 
        hints_hint = ""
        if task.hints_text:
            hints_hint = f"\n\nHints from issue tracker:\n{task.hints_text[:400]}"
 
        system = (
            "You are an expert software engineer fixing a real GitHub issue.\n"
            f"Repository: {task.repo}\n"
            f"Commit: {task.base_commit or 'HEAD'}\n"
            f"{file_context}\n"
            "\n"
            "RULES:\n"
            "1. Output ONLY a valid unified diff starting exactly with: diff --git\n"
            "2. Use EXACT line content from the files shown above — do NOT invent lines\n"
            "3. Include 3 lines of unchanged context before and after each change\n"
            "4. Do NOT include <think> tags, explanations, or markdown fences\n"
            "5. Make the MINIMAL change that fixes the described bug\n"
            "6. If multiple files need changes, include all diffs in one output\n"
            "\n"
            "DIFF FORMAT:\n"
            "diff --git a/path/to/file.ext b/path/to/file.ext\n"
            "--- a/path/to/file.ext\n"
            "+++ b/path/to/file.ext\n"
            "@@ -LINE,COUNT +LINE,COUNT @@\n"
            " context line\n"
            "-removed line\n"
            "+added line\n"
            " context line\n"
        )
 
        user = f"Fix this GitHub issue:\n\n{task.problem_statement[:2500]}{test_hint}{hints_hint}"
 
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]


    def _force_to_patch(self, raw: str, session: dict) -> dict:
        """Parse LLM output and force it into patch format."""
        raw = raw.strip()
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        # Strip markdown fences
        m = None
        if "```" in raw:
           m = re.search(r"```(?:diff|patch)?\n(.*?)```", raw, re.DOTALL)
        if m:
           raw = m.group(1).strip()
        # Find diff block
        if not (raw.startswith("diff --git") or raw.startswith("--- ")):
           m = re.search(r"(diff --git.*)", raw, re.DOTALL)
           raw = m.group(1).strip() if m else ""
        if raw:
           logger.info("[%s] Valid patch extracted (%d chars)", session["id"][:20], len(raw))
        else:
           logger.warning("[%s] No valid diff found in LLM output", session["id"][:20])
        return {"action": "patch", "content": raw}

    # ── Prompt Construction ───────────────────────────────────────────────────

    def _build_messages(self, session: dict) -> list[dict]:
        task    = session["task"]
        history = session["history"]
        state   = session["node_state"]
        system  = "You are a software engineer. Fix the bug described."
        summary = state.summarize()
        if summary:
            system += f"\n\nCurrent state:\n{summary}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": task.problem_statement[:2000]},
        ]
        for entry in history:
            action = entry.get("action", {})
            messages.append({
                "role":    "assistant",
                "content": f"<command>{action.get('action','bash')}</command>\n<content>{action.get('content','')}</content>",
            })
            obs = entry.get("observation")
            if obs:
                messages.append({"role": "user", "content": str(obs)[:1000]})
        return messages

    # ── State Update ──────────────────────────────────────────────────────────

    def _update_state(self, state: NodeState, obs: dict):
        state.cwd = obs.get("cwd", state.cwd)
        stdout    = obs.get("stdout", "")
        for fpath, lineno in re.findall(r'([\w/.-]+\.py)(?::(\d+))?', stdout)[:8]:
            state.add_file(fpath)
            if lineno:
                state.discovery_log[f"{fpath}:{lineno}"] = "referenced in output"

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_action(self, raw: str, session: dict) -> dict:
        raw     = raw.strip()
        command = (self._tag(raw, "command") or "bash").strip().lower()
        content = (self._tag(raw, "content") or "").strip()
        if command not in ("bash", "debug", "patch"):
            command = "patch" if ("diff --git" in content or "--- a/" in content) else "bash"
        return {"action": command, "content": content}
 

    @staticmethod
    def _tag(text: str, tag: str) -> str | None:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return m.group(1).strip() if m else None

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        lines = []
        for i, entry in enumerate(history, 1):
            act = entry.get("action", {})
            obs = entry.get("observation", {})
            lines.append(f"Step {i}: [{act.get('action')}] {act.get('content','')[:80]}")
            if obs:
                lines.append(f"  → {obs.get('stdout','')[:150]}")
        return "\n".join(lines)


# ==============================================================================
# FASTAPI APP
# ==============================================================================

app   = FastAPI(title="Purple Coding Agent")
agent = PurpleAgent()

AGENT_CARD = {
    "name": "Purple Coding Agent",
    "description": (
        "MCTS-guided software engineering agent for SWE-bench Pro. "
        "Powered by DeepSeek V3.2 + GitHub context."
    ),
    "url": f"http://localhost:{PORT}/",
    "version": "1.0.0",
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
            "description": "Explore a codebase and return a unified git diff patch.",
            "tags": ["coding", "swe-bench", "patch"],
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
    return {"status": "ok", "agent": "purple-coding-agent"}


@app.post("/")
async def handle_task(request: Request):
    body = await request.json()
 
    jsonrpc_id  = body.get("id", str(uuid.uuid4()))
    task_id     = str(uuid.uuid4())
    artifact_id = str(uuid.uuid4())
 
    logger.info("─" * 50)
    logger.info("Request id=%s method=%s", jsonrpc_id, body.get("method"))
 
    task_data, context_id = _extract_task_and_context(body)
    if not context_id:
        context_id = str(uuid.uuid4())
 
    task_data["session_id"] = context_id
 
    ps = task_data.get("problem_statement", "")
    logger.info("context_id=%s ps_len=%d repo=%s commit=%s",
                context_id[:20], len(ps),
                task_data.get("repo", "?"),
                task_data.get("base_commit", "")[:12] or "HEAD")
 
    action        = agent.respond(task_data)
    artifact_text = action.get("content", "") if action.get("action") == "patch" else json.dumps(action)
 
    logger.info("Response: action=%s artifact_len=%d", action.get("action"), len(artifact_text))
 
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
    context_id = ""
 
    if "problem_statement" in body:
        return body, context_id
 
    try:
        params     = body.get("params", {})
        message    = params.get("message", {})
        context_id = message.get("contextId", "") or params.get("contextId", "")
 
        parts = message.get("parts", [])
        logger.info("Parts: %d  contextId=%s", len(parts), context_id[:20] if context_id else "none")
 
        for i, part in enumerate(parts):
            kind = part.get("kind") or part.get("type", "")
            text = part.get("text", "")
            logger.info("Part[%d] kind=%s len=%d preview=%s", i, kind, len(text), text[:200])
 
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
                        if any(k in parsed for k in ("stdout", "stderr", "cwd", "repo", "instance_id")):
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