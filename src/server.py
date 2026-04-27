"""
Purple Coding Agent — Complete Server
======================================
Key fixes in this version:
  1. _extract_relevant_window: REMOVED line number prefixes (were breaking git apply)
  2. fetch_relevant_files: test paths used only as HINTS, not fetched directly
     Source files derived from test paths; tree search excludes cypress/e2e/stories
  3. System prompt: removed "use line numbers" instruction (no numbers shown now)
  4. Window stays at 80 lines but raw — model can copy context lines exactly
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
MODEL_NAME    = os.getenv("MODEL_NAME", "deepseek/deepseek-v3.2")
API_KEY       = (
    os.getenv("OPENROUTER_API_KEY", "")
    or os.getenv("LLM_API_KEY", "")
    or os.getenv("HF_TOKEN", "")
)
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")
PORT          = int(os.getenv("PORT", "9010"))
MAX_TURNS     = int(os.getenv("MAX_TURNS", "10"))
MCTS_BRANCHES = int(os.getenv("MCTS_BRANCHES", "6"))
TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.6"))
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
logger.info("MCTS          branches=%d  max_turns=%d  enabled=%s",
            MCTS_BRANCHES, MAX_TURNS, USE_MCTS)
logger.info("=" * 60)
logger.info("requirements=%s interface=%s",
    str(message.get("requirements", ""))[:100],
    str(message.get("interface", ""))[:100])


# ==============================================================================
# GITHUB FILE FETCHING
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
                logger.info("GitHub raw fetch OK: %s (%d chars)", filepath, len(content))
                return content
    except urllib.error.HTTPError as e:
        logger.warning("GitHub fetch failed for %s: HTTP %d", filepath, e.code)
    except Exception as e:
        logger.warning("GitHub fetch failed for %s: %s", filepath, e)
    return ""


def get_repo_tree(repo: str, ref: str) -> list[str]:
    """Get all file paths in repo using Git Tree API."""
    ref = ref or "HEAD"
    url = f"https://api.github.com/repos/{repo}/git/trees/{ref}?recursive=1"
    try:
        req = urllib.request.Request(url, headers=_github_headers())
        with urllib.request.urlopen(req, timeout=15) as r:
            data  = json.loads(r.read().decode())
            paths = [item["path"] for item in data.get("tree", [])
                     if item["type"] == "blob"]
            logger.info("Repo tree: %d files in %s", len(paths), repo)
            return paths
    except Exception as e:
        logger.warning("Tree API failed for %s: %s", repo, e)
        return []


# Directories and filename patterns to ALWAYS exclude when guessing source files
_EXCLUDE_DIRS = {
    "test", "tests", "spec", "specs", "mock", "mocks",
    "vendor", "node_modules", "__pycache__", ".git",
    "cypress", "e2e", "fixtures", "storybook", "stories",
    "__tests__", "snapshots", "testdata", "testutil",
    "example", "examples", "docs", "doc", "dist", "build",
}

_EXCLUDE_PATTERNS = re.compile(
    r'(test|spec|mock|fixture|story|stories|e2e|cypress|snapshot|'
    r'__tests__|testutil|testdata|_test)\.(py|go|js|ts|tsx|jsx)$',
    re.IGNORECASE,
)


def _is_source_file(path: str) -> bool:
    """Return True if the path looks like a source file (not test/fixture)."""
    parts = path.lower().split("/")
    # Exclude if any directory component is in exclude list
    for part in parts[:-1]:
        if part in _EXCLUDE_DIRS:
            return False
    # Exclude if filename matches test patterns
    filename = parts[-1]
    if _EXCLUDE_PATTERNS.search(filename):
        return False
    return True


def _derive_source_paths_from_test(test_path: str) -> list[str]:
    """
    Given a test file path, guess the corresponding source file paths.
    e.g. tests/test_foo.py → [foo.py, src/foo.py, lib/foo.py]
         scanner/redhat_test.go → [scanner/redhat.go]
         __tests__/Foo.test.ts → [Foo.ts, src/Foo.ts, components/Foo.ts]
    """
    candidates = []
    dirname  = "/".join(test_path.split("/")[:-1])
    filename = test_path.split("/")[-1]

    # Go style: foo_test.go → foo.go (same directory)
    if filename.endswith("_test.go"):
        src_name = filename[:-len("_test.go")] + ".go"
        candidates.append(f"{dirname}/{src_name}" if dirname else src_name)

    # Python style: test_foo.py → foo.py
    elif filename.startswith("test_") and filename.endswith(".py"):
        src_name = filename[len("test_"):]
        candidates.append(f"{dirname}/{src_name}" if dirname else src_name)
        candidates.append(f"src/{src_name}")

    # JS/TS style: Foo.test.ts → Foo.ts, Foo.spec.tsx → Foo.tsx
    elif re.search(r'\.(test|spec)\.(ts|tsx|js|jsx)$', filename):
        src_name = re.sub(r'\.(test|spec)(\.(ts|tsx|js|jsx))$', r'\2', filename)
        if dirname:
            # Remove test directory from path
            src_dir = re.sub(r'(^|/)(__tests__|tests?|specs?)(/|$)', '/', dirname)
            src_dir = src_dir.strip("/")
            candidates.append(f"{src_dir}/{src_name}" if src_dir else src_name)
        candidates.append(f"src/{src_name}")

    return [c for c in candidates if c]


def fetch_relevant_files(task: "SWETask") -> dict[str, str]:
    """
    Fetch actual file content from GitHub.

    Strategy:
    0. If fail_to_pass provided: fetch those test files directly
    1. Full paths explicitly mentioned in problem statement
    2. Derive source paths from test paths (test_foo.py → foo.py)
    3. Tree API + keyword ranking for SOURCE files
    4. If fail_to_pass is EMPTY (green agent withholds them):
       use tree API to find the most relevant TEST files by keyword match
    """
    if not task.repo:
        return {}

    ref   = task.base_commit or "HEAD"
    files: dict[str, str] = {}

    ps_words = set(re.findall(r'\b\w{4,}\b', task.problem_statement.lower()))

    def path_score(p: str) -> int:
        p_words = set(re.findall(r'\b\w{4,}\b', p.lower()))
        return len(ps_words & p_words)

    # ── Step 0: fetch failing test files if provided ──────────────────────────
    # Green agent often withholds fail_to_pass, so this may be empty
    for test in task.fail_to_pass[:4]:
        test_file = test.split("::")[0]
        if not test_file:
            continue
        # Only try if it looks like a file path (has / or known extension)
        looks_like_path = "/" in test_file or bool(
            re.search(r'\.(py|go|js|ts|tsx|jsx|java|rb|rs)$', test_file)
        )
        if not looks_like_path:
            logger.warning("fail_to_pass '%s' is not a file path — skipping", test_file)
            continue
        if test_file not in files:
            content = fetch_file_raw(task.repo, ref, test_file)
            if content:
                files[f"[FAILING TEST] {test_file}"] = content
                logger.info("Fetched failing test: %s", test_file)
            else:
                logger.warning("Test file 404: %s", test_file)

    # ── Step 1: full paths from problem statement ─────────────────────────────
    ps_full_paths = re.findall(
        r'(?:^|[\s`"\'(])('
        r'[\w][\w/.-]+\.(?:py|go|js|ts|tsx|jsx|java|rb|rs|c|cpp|h|php|cs|swift|kt)'
        r')',
        task.problem_statement,
        re.MULTILINE,
    )
    for fp in [p.strip() for p in ps_full_paths][:5]:
        if fp and len(files) < 6 and _is_source_file(fp):
            content = fetch_file_raw(task.repo, ref, fp)
            if content:
                files[fp] = content

    # ── Step 2: derive source paths from test paths ───────────────────────────
    for test in task.fail_to_pass[:4]:
        test_file = test.split("::")[0]
        derived   = _derive_source_paths_from_test(test_file)
        for fp in derived:
            if len(files) >= 6:
                break
            if fp not in files:
                content = fetch_file_raw(task.repo, ref, fp)
                if content:
                    files[fp] = content

    # ── Step 3: tree API for source files ────────────────────────────────────
    source_count = sum(1 for k in files if not k.startswith("[FAILING TEST]"))
    if source_count < 2:
        tree = get_repo_tree(task.repo, ref)
        if tree:
            src_files = [
                p for p in tree
                if re.search(r'\.(py|go|js|ts|tsx|jsx|rb|java|rs|c|cpp|h)$', p)
                and _is_source_file(p)
            ]
            ranked_src = sorted(src_files, key=path_score, reverse=True)
            for fp in ranked_src[:6]:
                if len(files) >= 8:
                    break
                if fp not in files:
                    content = fetch_file_raw(task.repo, ref, fp)
                    if content:
                        files[fp] = content

            # ── Step 4: when fail_to_pass is empty, also find relevant TEST files
            # The green agent withholds test names, so we find them via keyword search
            if not task.fail_to_pass:
                test_files = [
                    p for p in tree
                    if re.search(r'\.(py|go|js|ts|tsx|jsx|rb|java|rs)$', p)
                    and not _is_source_file(p)  # IS a test file
                ]
                ranked_tests = sorted(test_files, key=path_score, reverse=True)
                for fp in ranked_tests[:2]:
                    if len(files) >= 10:
                        break
                    if fp not in files:
                        content = fetch_file_raw(task.repo, ref, fp)
                        if content:
                            files[f"[RELEVANT TEST] {fp}"] = content
                            logger.info("Fetched relevant test by keyword: %s", fp)

    logger.info("Fetched %d files from GitHub for %s", len(files), task.repo)
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

    def is_leaf(self) -> bool:
        return not self.children


class MCTSEngine:
    def __init__(self, root: MCTSNode, branches: int = 6):
        self.root     = root
        self.branches = branches
        self._current = root
        self._pending: MCTSNode | None = None

    def select_action(self, candidates: list[tuple[dict, float]]) -> tuple[dict, MCTSNode]:
        for action, score in candidates:
            child = MCTSNode(state=self._current.state.copy(),
                             parent=self._current, action=action)
            child.update(score)
            self._current.children.append(child)
        best = self._current.best_child()
        if best is None:
            return candidates[0][0], self._current
        self._pending = best
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
            "nodes":      self._count(self.root),
            "depth":      self._depth(self._current),
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
# PROGRAMMABLE PRM
# ==============================================================================

class ProgrammablePRM:
    def score_static(self, action: dict, task: "SWETask") -> float:
        a = action.get("action", "bash")
        c = action.get("content", "")
        return min(0.20 * self._format(a, c) + 0.35 * self._relevance(c, task), 1.0)

    def score_observation(self, obs: dict, task: "SWETask") -> float:
        stdout = obs.get("stdout", "")
        stderr = obs.get("stderr", "")
        score  = 0.30
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
            return 0.4 * has_header + 0.3 * has_hunk + 0.3 * has_change
        return 1.0 if len(content) < 500 else 0.5

    def _relevance(self, content: str, task: "SWETask") -> float:
        if not task.problem_statement:
            return 0.5
        ps_tok = set(re.findall(r"\b\w{4,}\b", task.problem_statement.lower()))
        ct_tok = set(re.findall(r"\b\w{4,}\b", content.lower()))
        if not ps_tok:
            return 0.5
        return min(len(ps_tok & ct_tok) / len(ps_tok) * 2, 1.0)

    def _exec(self, stdout: str, stderr: str, task: "SWETask") -> float:
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

    def _discovery(self, stdout: str, task: "SWETask") -> float:
        files = re.findall(r'[\w/.-]+\.(?:py|go|ts|tsx|js)(?::\d+)?', stdout)
        if not files:
            return 0.0
        ps_lower = task.problem_statement.lower()
        for fp in files:
            base = re.split(r'[/.]', fp)[-2] if '.' in fp else fp
            if base.lower() in ps_lower:
                return 1.0
        return 0.5


# ==============================================================================
# NODE STATE
# ==============================================================================

@dataclass
class NodeState:
    cwd: str = "/workspace/repo"
    working_set: list[str] = field(default_factory=list)
    discovery_log: dict[str, str] = field(default_factory=dict)
    current_patch: str = ""

    def copy(self) -> "NodeState":
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
            for loc, fact in list(self.discovery_log.items())[-5:]:
                parts.append(f"  • {loc}: {fact}")
        return "\n".join(parts) if parts else ""


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
    requirements: str = ""
    interface: str = ""


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
        temperature: float = 0.6,
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
                    logger.error("LLM 401 — check OPENROUTER_API_KEY secret")
                    return ""
                if resp.status_code == 429:
                    time.sleep(5 * attempt)
                    continue
                if resp.status_code != 200:
                    logger.error("LLM %d: %s", resp.status_code, resp.text[:300])
                    return ""
                content = resp.json()["choices"][0]["message"]["content"]
                logger.info("LLM status=200 returned %d chars", len(content))
                return content
            except requests.RequestException as e:
                logger.warning("LLM attempt %d/3: %s", attempt, e)
                if attempt < 3:
                    time.sleep(2 ** attempt)
                else:
                    return ""
        return ""


# ==============================================================================
# PURPLE AGENT
# ==============================================================================

class PurpleAgent:
    def __init__(self):
        self.llm      = LLMClient()
        self.prm      = ProgrammablePRM()
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
            return self._step_sync(session, message)
        except Exception as e:
            logger.exception("[%s] Agent step crashed: %s", session_id, e)
            return {"action": "patch", "content": ""}

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
            base_commit=message.get("base_commit", ""),
            requirements=message.get("requirements", ""),
            interface=message.get("interface", ""),
        )
        root    = MCTSNode(state={"cwd": task.cwd})
        session = {
            "id":              session_id,
            "task":            task,
            "mcts":            MCTSEngine(root, branches=MCTS_BRANCHES),
            "node_state":      NodeState(cwd=task.cwd),
            "turn":            0,
            "history":         [],
            "submitted_patch": None,
            "_fetched_files":  None,
        }
        self._sessions[session_id] = session
        logger.info("[%s] requirements_len=%d interface_len=%d",
            session_id[:20],
            len(task.requirements),
            len(task.interface)) 
        logger.info("[%s] New session repo=%s commit=%s",
                    session_id[:20], task.repo,
                    task.base_commit[:12] if task.base_commit else "HEAD")
        return session

    def _step_sync(self, session: dict, message: dict) -> dict:
        """Synchronous entry point — runs async _step in event loop."""
        return asyncio.run(self._step(session, message))

    async def _step(self, session: dict, message: dict) -> dict:
        task: SWETask = session["task"]
        session["turn"] += 1
        turn = session["turn"]

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
                self._update_state(session["node_state"], obs)
                logger.info("[%s] turn=%d PRM=%.3f", session["id"][:20], turn, score)

        if turn >= MAX_TURNS - 1 and not session["submitted_patch"]:
            logger.info("[%s] Final turn — forcing patch", session["id"][:20])
            return self._force_patch(session)

        if USE_MCTS:
            action = await self._mcts_patch(session)
        else:
            action = self._greedy_patch(session)

        session["history"].append({"action": action})
        logger.info("[%s] turn=%d action=%s", session["id"][:20], turn, action.get("action"))
        return action

    def _greedy_patch(self, session: dict) -> dict:
        msgs = self._build_patch_messages(session)
        raw  = self.llm.complete(msgs, temperature=0.2, max_tokens=2048)
        return self._force_to_patch(raw, session)

    async def _mcts_patch(self, session: dict) -> dict:
        msgs = self._build_patch_messages(session)

        tasks = [
            asyncio.to_thread(
                self.llm.complete, msgs,
                temperature=0.2 + i * 0.15,
                max_tokens=2048,
            )
            for i in range(MCTS_BRANCHES)
        ]
        raws = await asyncio.gather(*tasks)

        candidates = []
        for i, raw in enumerate(raws):
            action = self._force_to_patch(raw, session)
            score  = self.prm.score_static(action, session["task"])
            candidates.append((action, score))
            logger.info("[%s] MCTS branch %d/%d score=%.3f patch_len=%d",
                        session["id"][:20], i+1, MCTS_BRANCHES,
                        score, len(action.get("content", "")))

        best_action, best_score = max(candidates, key=lambda x: x[1])
        logger.info("[%s] MCTS selected score=%.3f", session["id"][:20], best_score)
        session["mcts"].backpropagate(best_score)
        return best_action

    def _force_patch(self, session: dict) -> dict:
        task    = session["task"]
        history = self._format_history(session["history"])
        msgs    = [
            {"role": "system", "content":
             "You are an expert software engineer. "
             "Output ONLY a unified diff starting with diff --git. No explanation."},
            {"role": "user", "content":
             f"Exploration so far:\n{history}\n\nFix:\n{task.problem_statement[:600]}"},
        ]
        raw    = self.llm.complete(msgs, temperature=0.15, max_tokens=2048)
        action = self._force_to_patch(raw, session)
        if action.get("action") != "patch":
            action = {"action": "patch", "content": ""}
        session["submitted_patch"] = action["content"]
        return action

    def _build_patch_messages(self, session: dict) -> list[dict]:
        task = session["task"]

        # Fetch files once per session (cached)
        if session["_fetched_files"] is None:
            session["_fetched_files"] = fetch_relevant_files(task)
        real_files = session["_fetched_files"]

        if real_files:
            file_sections = []
            for filepath, content in real_files.items():
                # FIX: NO LINE NUMBERS — show raw content only
                # Line numbers were being copied into diff context lines,
                # causing every git apply to fail with context mismatch.
                displayed = self._extract_relevant_window(content, task, window=80)
                file_sections.append(f"### {filepath}\n```\n{displayed}\n```")
            file_context = (
                "\n\n## ACTUAL FILE CONTENTS\n"
                "Copy context lines EXACTLY as shown — character for character:\n\n"
                + "\n\n".join(file_sections)
            )
            logger.info("[%s] Providing %d real files to model",
                        session["id"][:20], len(real_files))
        else:
            file_context = (
                "\n\n## NOTE\n"
                "File content unavailable. Infer from the issue and test paths."
            )
            logger.warning("[%s] No files fetched — model working blind",
                           session["id"][:20])

        test_hint = ""
        if task.fail_to_pass:
            test_hint = "\n\nFailing tests:\n" + "\n".join(
                f"  - {t}" for t in task.fail_to_pass[:5]
            )

        hints_hint = ""
        if task.hints_text:
            hints_hint = f"\n\nHints:\n{task.hints_text[:400]}"

        system = (
            "You are an expert software engineer fixing a real GitHub issue.\n"
            f"Repository: {task.repo}\n"
            f"Commit: {task.base_commit or 'HEAD'}\n"
            f"{file_context}\n"
            "\n"
            "STRATEGY:\n"
            "1. Read [FAILING TEST] or [RELEVANT TEST] files first — they show what behavior is tested\n"
            "2. Understand exactly what the test expects the source code to do\n"
            "3. Find the source code responsible for that behavior in the other files\n"
            "4. Make the MINIMAL change to the SOURCE code to satisfy the test\n"
            "5. Do NOT modify any test file — only modify source files\n"
            "\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY a valid unified diff starting exactly with: diff --git\n"
            "2. Copy context lines CHARACTER-FOR-CHARACTER from the files above\n"
            "   DO NOT paraphrase or reformat context lines even slightly\n"
            "   One space difference will break git apply\n"
            "3. The patch may span MULTIPLE FILES and 50-150 lines — implement everything required\n"
            "4. Include exactly 3 unchanged context lines before and after changes\n"
            "5. Do NOT touch [FAILING TEST] or [RELEVANT TEST] files — diff source files only\n"
            "6. Do NOT include <think> tags, explanations, or markdown fences\n"
            "\n"
            "DIFF FORMAT:\n"
            "diff --git a/path/file.go b/path/file.go\n"
            "--- a/path/file.go\n"
            "+++ b/path/file.go\n"
            "@@ -42,7 +42,7 @@\n"
            " exact context line from file\n"
            " exact context line from file\n"
            " exact context line from file\n"
            "-old line to remove\n"
            "+new line to add\n"
            " exact context line from file\n"
            " exact context line from file\n"
            " exact context line from file\n"
        )

        user = (
            f"## Problem Statement\n\n{task.problem_statement[:2000]}\n\n"
        )
        if task.requirements:
            user += f"## Requirements\n(These are grounded on the tests — implement ALL of these)\n\n{task.requirements}\n\n"
        if task.interface:
            user += f"## Interface\n(Use EXACTLY these function/class names and signatures)\n\n{task.interface}\n\n"
        user += test_hint + hints_hint

        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    def _extract_relevant_window(self, content: str, task: SWETask,
                                  window: int = 80) -> str:
        """
        Find the most relevant window of lines using keyword matching.

        FIX: Returns RAW lines with NO line number prefixes.
        Previous version added '  42: ' prefix to every line, which the model
        then copied into diff context lines, making git apply fail 100% of the time.
        """
        lines = content.splitlines()
        if len(lines) <= window:
            # File fits entirely — show all of it, raw
            return content

        # Keywords from problem statement + test function names
        keywords = set(re.findall(r'\b\w{4,}\b', task.problem_statement.lower()))
        for test in task.fail_to_pass:
            func = test.split("::")[-1] if "::" in test else ""
            if func:
                keywords.update(re.findall(r'\b\w{4,}\b', func.lower()))

        # Score each line by keyword density
        scores = [
            len(keywords & set(re.findall(r'\b\w{4,}\b', line.lower())))
            for line in lines
        ]

        # Find the window with highest total keyword score
        best_start = 0
        best_score = -1
        step = max(1, len(lines) // 20)  # check ~20 positions
        for start in range(0, max(1, len(lines) - window), step):
            ws = sum(scores[start:start + window])
            if ws > best_score:
                best_score = ws
                best_start = start

        selected = lines[best_start:best_start + window]

        # Header shows line range for reference (NOT prepended to each line)
        header = f"[Lines {best_start+1}-{best_start+len(selected)} of {len(lines)} total]\n"
        return header + "\n".join(selected)

    def _force_to_patch(self, raw: str, session: dict) -> dict:
        raw = raw.strip()
        # Strip DeepSeek <think> blocks
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
            logger.info("[%s] Valid patch extracted (%d chars)",
                        session["id"][:20], len(raw))
        else:
            logger.warning("[%s] No valid diff in LLM output", session["id"][:20])
        return {"action": "patch", "content": raw}

    def _update_state(self, state: NodeState, obs: dict):
        state.cwd = obs.get("cwd", state.cwd)
        stdout    = obs.get("stdout", "")
        for fpath, lineno in re.findall(
            r'([\w/.-]+\.(?:py|go|ts|tsx|js))(?::(\d+))?', stdout
        )[:8]:
            state.add_file(fpath)
            if lineno:
                state.discovery_log[f"{fpath}:{lineno}"] = "referenced in output"

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
    "description": "MCTS-guided SWE-bench agent. DeepSeek V3 + GitHub context.",
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
            "description": "Fix a GitHub issue and return a unified git diff patch.",
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
                (task_data.get("base_commit", "") or "")[:12] or "HEAD")

    # Run agent — respond() calls asyncio.run() internally for _mcts_patch
    import concurrent.futures
    loop   = asyncio.get_event_loop()
    action = await loop.run_in_executor(None, agent.respond, task_data)

    artifact_text = (
        action.get("content", "")
        if action.get("action") == "patch"
        else json.dumps(action)
    )

    logger.info("Response: action=%s artifact_len=%d",
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
                        i, kind, len(text), text[:200])

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