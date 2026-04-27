"""
Purple Coding Agent — Two-Stage Pipeline
==========================================
Stage 1: LLM LOCALIZATION
  - Fetch repo tree from GitHub
  - Ask LLM: "Given this problem, which files need to change?"
  - LLM returns JSON list of file paths with reasoning
  - Fetch those exact files

Stage 2: MCTS REPAIR
  - 6 parallel branches, each with actual file content
  - PRM scores by patch validity
  - Return highest-scoring patch

Why this beats keyword search:
  LLMs understand semantics — "caching bug in cluster config →
  look in lib/services/ not lib/client/" — keyword matching cannot.

Facts established from logs:
  - Green agent sends: problem_statement, repo, base_commit only
  - fail_to_pass, requirements, interface are always empty
  - pass_to_pass_ok always True (patches apply)
  - fail_to_pass_ok always False (wrong code patched)
  - Root cause: we were fetching wrong files via keyword guessing
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
import urllib.parse
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
logger.info("MCTS          branches=%d  enabled=%s", MCTS_BRANCHES, USE_MCTS)
logger.info("=" * 60)


# ==============================================================================
# GITHUB HELPERS
# ==============================================================================

def _github_headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def fetch_file_raw(repo: str, ref: str, filepath: str) -> str:
    """Fetch raw file content from raw.githubusercontent.com."""
    ref = ref or "HEAD"
    url = f"https://raw.githubusercontent.com/{repo}/{ref}/{filepath}"
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            if r.status == 200:
                content = r.read().decode("utf-8", errors="replace")
                logger.info("Fetched: %s (%d chars)", filepath, len(content))
                return content
    except urllib.error.HTTPError as e:
        logger.warning("GitHub 404: %s (HTTP %d)", filepath, e.code)
    except Exception as e:
        logger.warning("GitHub error: %s — %s", filepath, e)
    return ""


def get_repo_tree(repo: str, ref: str) -> list[str]:
    """Get all file paths using GitHub Git Tree API (recursive)."""
    ref = ref or "HEAD"
    url = f"https://api.github.com/repos/{repo}/git/trees/{ref}?recursive=1"
    try:
        req = urllib.request.Request(url, headers=_github_headers())
        with urllib.request.urlopen(req, timeout=20) as r:
            data  = json.loads(r.read().decode())
            paths = [item["path"] for item in data.get("tree", [])
                     if item["type"] == "blob"]
            logger.info("Repo tree: %d files in %s", len(paths), repo)
            return paths
    except Exception as e:
        logger.warning("Tree API failed for %s: %s", repo, e)
        return []


# ==============================================================================
# STAGE 1 — LLM LOCALIZATION
# ==============================================================================

_SOURCE_EXTS = re.compile(
    r'\.(py|go|js|ts|tsx|jsx|java|rb|rs|c|cpp|h|php|cs|swift|kt|vue|svelte)$'
)

_EXCLUDE_DIRS = {
    "vendor", "node_modules", "__pycache__", ".git", "dist", "build",
    "testdata", "fixtures", "migrations", "generated", "proto",
}


def _filter_tree_for_llm(paths: list[str], max_paths: int = 300) -> list[str]:
    """
    Filter the repo tree to source files only, excluding generated/vendor dirs.
    Cap at max_paths to fit in LLM context.
    """
    filtered = []
    for p in paths:
        parts = p.split("/")
        if any(part in _EXCLUDE_DIRS for part in parts):
            continue
        if _SOURCE_EXTS.search(p):
            filtered.append(p)
    # If still too many, prefer shorter paths (top-level src more likely relevant)
    filtered.sort(key=lambda p: (len(p.split("/")), p))
    return filtered[:max_paths]


def llm_localize(
    problem_statement: str,
    repo: str,
    tree_paths: list[str],
    llm_client: "LLMClient",
) -> list[str]:
    """
    Stage 1: Ask the LLM which files in the repo need to change.

    Returns a list of file paths (up to 6) that the LLM identifies as
    needing modification to fix the described bug/feature.
    """
    if not tree_paths:
        return []

    filtered_paths = _filter_tree_for_llm(tree_paths, max_paths=300)
    tree_str = "\n".join(filtered_paths)

    system = (
        "You are a senior software engineer performing code localization.\n"
        "Given a bug report and the repository file tree, identify which source files "
        "need to be modified to fix the issue.\n"
        "\n"
        "RULES:\n"
        "1. Return ONLY a JSON array of file paths, nothing else\n"
        "2. Include 3-6 files maximum — the ones most likely to contain the bug\n"
        "3. Prefer source files over test files\n"
        "4. Use EXACT paths from the file tree provided\n"
        "5. Think about which module/package the bug would live in based on the description\n"
        "\n"
        'Example output: ["lib/config/cluster.go", "lib/services/auth.go"]'
    )

    user = (
        f"Repository: {repo}\n\n"
        f"Bug Report:\n{problem_statement[:3000]}\n\n"
        f"Repository File Tree:\n{tree_str}\n\n"
        "Which files need to be modified to fix this? Return a JSON array of file paths."
    )

    raw = llm_client.complete(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
        temperature=0.1,   # Low temp for localization — we want deterministic reasoning
        max_tokens=512,
    )

    if not raw:
        logger.warning("Localization LLM call returned empty")
        return []

    # Strip <think> blocks from reasoning models
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # Extract JSON array from response
    try:
        # Try direct parse first
        paths = json.loads(raw)
        if isinstance(paths, list):
            valid = [p for p in paths if isinstance(p, str) and p in set(tree_paths)]
            logger.info("Localization found %d valid paths: %s", len(valid), valid)
            return valid[:6]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting JSON array from within text
    m = re.search(r'\[.*?\]', raw, re.DOTALL)
    if m:
        try:
            paths = json.loads(m.group(0))
            if isinstance(paths, list):
                valid = [p for p in paths if isinstance(p, str) and p in set(tree_paths)]
                logger.info("Localization (extracted) found %d paths: %s", len(valid), valid)
                return valid[:6]
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning("Could not parse localization response: %s", raw[:200])
    return []


def fetch_located_files(
    task: "SWETask",
    located_paths: list[str],
) -> dict[str, str]:
    """
    Fetch the files identified by LLM localization.
    Falls back to keyword-based fetching if localization returns nothing.
    """
    ref   = task.base_commit or "HEAD"
    files: dict[str, str] = {}

    # Fetch LLM-located files
    for fp in located_paths:
        content = fetch_file_raw(task.repo, ref, fp)
        if content:
            files[fp] = content

    # Fallback: any explicitly named files in problem statement
    if len(files) < 2:
        ps_paths = re.findall(
            r'(?:^|[\s`"\'(])([\w][\w/.-]+\.(?:py|go|js|ts|tsx|jsx|java|rb|rs|c|cpp|h))',
            task.problem_statement,
            re.MULTILINE,
        )
        for fp in [p.strip() for p in ps_paths][:4]:
            if fp and fp not in files:
                content = fetch_file_raw(task.repo, ref, fp)
                if content:
                    files[fp] = content

    logger.info("Fetched %d files total for %s", len(files), task.repo)
    return files


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
        return max(self.children, key=lambda c: c.uct(self._visits)) if self.children else None


class MCTSEngine:
    def __init__(self, root: MCTSNode, branches: int = 3):
        self.root     = root
        self.branches = branches
        self._current = root
        self._pending: MCTSNode | None = None

    def select_action(self, candidates: list[tuple[dict, float]]) -> dict:
        for action, score in candidates:
            child = MCTSNode(state=self._current.state.copy(),
                             parent=self._current, action=action)
            child.update(score)
            self._current.children.append(child)
        best = self._current.best_child()
        if best is None:
            return candidates[0][0]
        self._pending = best
        return best.action

    def backpropagate(self, reward: float):
        node = self._pending or self._current
        while node is not None:
            node.update(reward)
            node = node.parent
        if self._pending:
            self._current = self._pending
            self._pending = None

    def stats(self) -> dict:
        return {"root_value": round(self.root.value, 3)}

    @staticmethod
    def _count(n: MCTSNode) -> int:
        return 1 + sum(MCTSEngine._count(c) for c in n.children)


# ==============================================================================
# PRM (PROCESS REWARD MODEL)
# ==============================================================================

class ProgrammablePRM:
    """Score a patch candidate without executing it."""

    def score(self, content: str, task: "SWETask") -> float:
        if not content.strip():
            return 0.0
        score = 0.0
        score += 0.30 * self._format(content)
        score += 0.40 * self._relevance(content, task)
        score += 0.30 * self._completeness(content)
        return min(score, 1.0)

    def _format(self, content: str) -> float:
        """Is it a valid unified diff?"""
        has_header = "diff --git" in content
        has_hunk   = "@@" in content
        has_minus  = bool(re.search(r'^-[^-]', content, re.MULTILINE))
        has_plus   = bool(re.search(r'^\+[^\+]', content, re.MULTILINE))
        has_lf     = "\n" in content
        return (0.3 * has_header + 0.2 * has_hunk +
                0.2 * has_minus + 0.2 * has_plus + 0.1 * has_lf)

    def _relevance(self, content: str, task: "SWETask") -> float:
        """Does the patch touch files/terms related to the problem?"""
        if not task.problem_statement:
            return 0.5
        ps_tok = set(re.findall(r'\b\w{4,}\b', task.problem_statement.lower()))
        ct_tok = set(re.findall(r'\b\w{4,}\b', content.lower()))
        if not ps_tok:
            return 0.5
        return min(len(ps_tok & ct_tok) / max(len(ps_tok) * 0.5, 1), 1.0)

    def _completeness(self, content: str) -> float:
        """Does the patch look complete (not truncated)?"""
        lines   = content.strip().splitlines()
        n_plus  = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
        n_minus = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
        if n_plus == 0 and n_minus == 0:
            return 0.0
        # Penalize empty patches or single-char patches
        if len(content.strip()) < 50:
            return 0.1
        return 1.0


# ==============================================================================
# TASK MODEL
# ==============================================================================

@dataclass
class SWETask:
    problem_statement: str
    cwd: str = "/workspace/repo"
    hints_text: str = ""
    repo: str = ""
    instance_id: str = ""
    base_commit: str = ""


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
        }
        for attempt in range(1, 4):
            try:
                resp = requests.post(
                    CHAT_URL, json=payload, headers=self._headers, timeout=90
                )
                logger.info("LLM status: %d", resp.status_code)
                if resp.status_code == 401:
                    logger.error("LLM 401 — check OPENROUTER_API_KEY")
                    return ""
                if resp.status_code == 429:
                    time.sleep(5 * attempt)
                    continue
                if resp.status_code != 200:
                    logger.error("LLM %d: %s", resp.status_code, resp.text[:300])
                    return ""
                content = resp.json()["choices"][0]["message"]["content"]
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
# STAGE 2 — PATCH GENERATION
# ==============================================================================

def _build_file_context(files: dict[str, str], task: SWETask) -> str:
    """Build the file context block for the repair prompt."""
    if not files:
        return "(No files fetched — patch based on problem description only)"

    sections = []
    for filepath, content in files.items():
        # Extract the most relevant 100-line window without line numbers
        window = _relevant_window(content, task, window=100)
        sections.append(f"### {filepath}\n```\n{window}\n```")

    return (
        "## SOURCE FILES (copy context lines CHARACTER-FOR-CHARACTER):\n\n"
        + "\n\n".join(sections)
    )


def _relevant_window(content: str, task: SWETask, window: int = 100) -> str:
    """Return the most bug-relevant window of lines from a file."""
    lines = content.splitlines()
    if len(lines) <= window:
        return content

    keywords = set(re.findall(r'\b\w{4,}\b', task.problem_statement.lower()))

    scores = [
        len(keywords & set(re.findall(r'\b\w{4,}\b', line.lower())))
        for line in lines
    ]

    best_start, best_score = 0, -1
    step = max(1, (len(lines) - window) // 20)
    for start in range(0, len(lines) - window + 1, step):
        ws = sum(scores[start:start + window])
        if ws > best_score:
            best_score, best_start = ws, start

    selected = lines[best_start:best_start + window]
    header   = f"[Lines {best_start+1}–{best_start+len(selected)} of {len(lines)}]\n"
    return header + "\n".join(selected)


def _build_repair_messages(
    task: SWETask,
    files: dict[str, str],
    located_paths: list[str],
) -> list[dict]:
    """Build the prompt for Stage 2 patch generation."""

    file_context = _build_file_context(files, task)

    located_hint = ""
    if located_paths:
        located_hint = (
            f"\n\nFiles identified as needing changes:\n"
            + "\n".join(f"  - {p}" for p in located_paths)
        )

    system = (
        "You are an expert software engineer fixing a real GitHub issue.\n"
        f"Repository: {task.repo}\n"
        f"Commit: {task.base_commit or 'HEAD'}\n"
        f"\n{file_context}\n"
        "\n"
        "RULES:\n"
        "1. Output ONLY a unified diff starting with: diff --git\n"
        "2. Copy context lines CHARACTER-FOR-CHARACTER from the files above\n"
        "   One character difference will break git apply\n"
        "3. The patch may span multiple files — include ALL necessary changes\n"
        "4. Include exactly 3 unchanged context lines before/after each change\n"
        "5. No <think> tags, no markdown fences, no explanation — raw diff only\n"
        "\n"
        "DIFF FORMAT:\n"
        "diff --git a/path/file.go b/path/file.go\n"
        "--- a/path/file.go\n"
        "+++ b/path/file.go\n"
        "@@ -42,7 +42,8 @@\n"
        " context\n"
        " context\n"
        " context\n"
        "-old line\n"
        "+new line\n"
        " context\n"
        " context\n"
        " context\n"
    )

    user = (
        f"## Problem\n\n{task.problem_statement[:2500]}"
        f"{located_hint}"
    )
    if task.hints_text:
        user += f"\n\n## Hints\n{task.hints_text[:400]}"

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def _extract_patch(raw: str) -> str:
    """Extract a valid unified diff from LLM output."""
    if not raw:
        return ""

    # Strip Gemma4 <think> blocks
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # Strip markdown fences
    if "```" in raw:
        m = re.search(r"```(?:diff|patch)?\n(.*?)```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()

    # Find diff block
    if not (raw.startswith("diff --git") or raw.startswith("--- ")):
        m = re.search(r"(diff --git.*)", raw, re.DOTALL)
        raw = m.group(1).strip() if m else ""

    return raw


# ==============================================================================
# PURPLE AGENT
# ==============================================================================

class PurpleAgent:
    """
    Two-stage agent:
      Stage 1: LLM localization (which files?)
      Stage 2: MCTS repair (3 parallel patch candidates, pick best)
    """

    def __init__(self):
        self.llm      = LLMClient()
        self.prm      = ProgrammablePRM()
        self.mcts     = MCTSEngine(MCTSNode(state={}), branches=MCTS_BRANCHES)
        self._sessions: dict[str, dict[str, Any]] = {}

    def respond(self, message: dict) -> dict:
        session_id = (
            message.get("session_id")
            or message.get("instance_id")
            or str(abs(hash(message.get("problem_statement", "")[:80])))
        )

        # Use cached session to avoid re-doing localization on repeat calls
        if session_id not in self._sessions:
            session = self._init_session(session_id, message)
        else:
            session = self._sessions[session_id]

        try:
            # asyncio.run is safe here — called from run_in_executor thread
            return asyncio.run(self._run(session))
        except Exception as e:
            logger.exception("[%s] Agent crashed: %s", session_id[:20], e)
            return {"action": "patch", "content": ""}

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
            "id":            session_id,
            "task":          task,
            "located_paths": None,  # set after localization
            "fetched_files": None,  # set after fetching
        }
        self._sessions[session_id] = session
        logger.info("[%s] New session repo=%s commit=%s",
                    session_id[:20], task.repo,
                    task.base_commit[:12] if task.base_commit else "HEAD")
        return session

    async def _run(self, session: dict) -> dict:
        task = session["task"]

        # ── Stage 1: Localization (once per session, cached) ─────────────────
        if session["located_paths"] is None:
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
                logger.info("[%s] Localization → %s",
                            session["id"][:20], located)
            else:
                session["located_paths"] = []
                logger.warning("[%s] Tree fetch failed — skipping localization",
                               session["id"][:20])

        # ── Fetch files (once per session, cached) ────────────────────────────
        if session["fetched_files"] is None:
            session["fetched_files"] = await asyncio.to_thread(
                fetch_located_files, task, session["located_paths"]
            )

        # ── Stage 2: MCTS patch generation (parallel branches) ───────────────
        if USE_MCTS:
            patch = await self._mcts_repair(session)
        else:
            patch = await self._greedy_repair(session)

        return {"action": "patch", "content": patch}

    async def _greedy_repair(self, session: dict) -> str:
        msgs = _build_repair_messages(
            session["task"],
            session["fetched_files"],
            session["located_paths"],
        )
        raw = await asyncio.to_thread(self.llm.complete, msgs, 0.2, 2048)
        return _extract_patch(raw)

    async def _mcts_repair(self, session: dict) -> str:
        """
        Run MCTS_BRANCHES patch candidates in parallel.
        Temperatures span 0.15–0.95 for diversity.
        Return the highest PRM-scoring valid patch.
        """
        task = session["task"]
        msgs = _build_repair_messages(
            task,
            session["fetched_files"],
            session["located_paths"],
        )

        # Temperature schedule: branch 0 is conservative, branch N-1 is creative
        temps = [
            0.15 + (0.80 / max(MCTS_BRANCHES - 1, 1)) * i
            for i in range(MCTS_BRANCHES)
        ]

        # Fire all branches simultaneously
        llm_tasks = [
            asyncio.to_thread(self.llm.complete, msgs, temp, 2048)
            for temp in temps
        ]
        raws = await asyncio.gather(*llm_tasks)

        candidates: list[tuple[str, float]] = []
        for i, raw in enumerate(raws):
            patch = _extract_patch(raw)
            score = self.prm.score(patch, task)
            candidates.append((patch, score))
            logger.info("[%s] Branch %d/%d T=%.2f score=%.3f len=%d",
                        session["id"][:20], i+1, MCTS_BRANCHES,
                        temps[i], score, len(patch))

        # Select best by PRM score
        best_patch, best_score = max(candidates, key=lambda x: x[1])

        # Update MCTS tree
        best_action = {"action": "patch", "content": best_patch}
        self.mcts.select_action([({"action": "patch", "content": p}, s)
                                  for p, s in candidates])
        self.mcts.backpropagate(best_score)

        logger.info("[%s] MCTS selected score=%.3f len=%d stats=%s",
                    session["id"][:20], best_score, len(best_patch),
                    self.mcts.stats())
        return best_patch


# ==============================================================================
# FASTAPI APP
# ==============================================================================

app   = FastAPI(title="Purple Coding Agent")
agent = PurpleAgent()

AGENT_CARD = {
    "name": "Purple Coding Agent",
    "description": (
        "Two-stage SWE-bench agent: LLM localization + MCTS repair. "
        "Gemma4-31B + GitHub API."
    ),
    "url": f"http://localhost:{PORT}/",
    "version": "2.0.0",
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
            "description": "Localize and fix a GitHub issue via unified diff patch.",
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
    return {"status": "ok", "agent": "purple-coding-agent", "version": "2.0.0"}


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
                (task_data.get("base_commit", "") or "")[:12] or "HEAD")

    # Run agent in thread pool so asyncio.run() inside works correctly
    loop   = asyncio.get_event_loop()
    action = await loop.run_in_executor(None, agent.respond, task_data)

    artifact_text = (
        action.get("content", "")
        if action.get("action") == "patch"
        else json.dumps(action)
    )

    logger.info("Response: action=%s len=%d",
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
                    if isinstance(parsed, dict) and "problem_statement" in parsed:
                        return parsed, context_id
                    if isinstance(parsed, dict) and any(
                        k in parsed for k in ("stdout", "stderr", "cwd", "repo", "instance_id")
                    ):
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