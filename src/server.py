"""
Purple Coding Agent — v4.1
================================================
5-Stage Architecture:

STAGE 1   — LLM LOCALIZATION
  GitHub Tree API → LLM → up to 5 file paths → fetch from GitHub

STAGE 1.5 — SYNTHETIC TEST FAILURE HYPOTHESES
  Given problem_statement + source files, LLM infers:
    * What assertion is likely failing?
    * Expected vs. actual values?
    * Root cause function/condition?
    * Concrete fix_hint?
  Returns 2-3 ranked hypotheses injected into every MCTS branch.

STAGE 2   — MCTS REPAIR (parallel generation)
  3 branches (T=0.15, 0.47, 0.80) each see bug report + hypotheses + files.
  Heuristic PRM scores format / relevance / completeness.

STAGE 2.5 — PLT SELF-CONSISTENCY CHECK
  ONE LLM call evaluates all 3 branches against the hypotheses:
    * Does each patch logic address the root_cause?
    * Is the programming technique sound?
  final_score = 0.40 * heuristic + 0.60 * plt_score
  MCTS selects the branch with highest final_score.

Green agent behavior (confirmed from logs):
  - Sends: problem_statement, repo, base_commit, instance_id
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
MODEL_NAME    = os.getenv("MODEL_NAME", "deepseek/deepseek-v4-flash")
API_KEY       = (
    os.getenv("OPENROUTER_API_KEY", "")
    or os.getenv("LLM_API_KEY", "")
    or os.getenv("HF_TOKEN", "")
)
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")
PORT          = int(os.getenv("PORT", "9010"))
MCTS_BRANCHES = int(os.getenv("MCTS_BRANCHES", "3"))
MCTS_ITERATIONS = int(os.getenv("MCTS_ITERATIONS", "3"))
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
logger.info("pipeline      MCTS_BRANCHES=%d + hypothesis synthesis", MCTS_BRANCHES)
logger.info("=" * 60)


# ==============================================================================
# GITHUB HELPERS  (Stage 1 only — tree + file fetch)
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
# STAGE 1.5 — SYNTHETIC TEST FAILURE HYPOTHESES  ← NEW
# ==============================================================================

def llm_synthesize_hypotheses(
    problem_statement: str,
    fetched_files: dict[str, str],
    llm: "LLMClient",
    n: int = 3,
) -> list[dict]:
    """
    Given the bug report and source files (no actual test files available),
    infer what automated tests are likely failing and WHY.

    This gives the MCTS repair branches a synthetic oracle:
    instead of blindly patching code, the model knows specifically what
    behaviour the test expects vs what the buggy code actually does.

    Returns a list of up to `n` hypothesis dicts, sorted by confidence desc:
      {
        "failure_mode":   str — which assertion/invariant is failing
        "expected_value": str — what the test expects the code to produce
        "actual_value":   str — what the buggy code currently produces
        "root_cause":     str — exact function / condition / missing branch at fault
        "fix_hint":       str — concrete, actionable description of the required change
        "confidence":     float — 0.0–1.0
      }
    """
    if not fetched_files:
        logger.info("Hypothesis synthesis skipped: no fetched files")
        return []

    # Build concise file context — first 120 lines per file to stay within budget
    file_sections = []
    for fp, content in fetched_files.items():
        lines = content.splitlines()
        snippet = "\n".join(lines[:120])
        if len(lines) > 120:
            snippet += f"\n# ... ({len(lines) - 120} more lines not shown)"
        file_sections.append(f"### {fp}\n```\n{snippet}\n```")
    file_ctx = "\n\n".join(file_sections)

    system = """\
You are an expert software engineer performing root-cause analysis.
You do NOT have access to the test files, but you can infer what they test
by reading the bug report carefully and tracing the source code.

Your goal: generate concrete hypotheses about which automated test assertions
are currently FAILING and what code change would make them PASS.

Return a JSON array of up to 3 hypotheses, sorted by confidence (highest first).
Each hypothesis is a JSON object with EXACTLY these keys:
  "failure_mode"  : (string) the specific assertion or invariant that fails — be precise
  "expected_value": (string) the value/behaviour the test expects the code to produce
  "actual_value"  : (string) the value/behaviour the buggy code actually produces
  "root_cause"    : (string) the specific function, branch, or condition that is wrong
  "fix_hint"      : (string) a concrete, actionable description of the exact code change needed
  "confidence"    : (float)  your confidence this is the real failure, 0.0–1.0

Requirements for good hypotheses:
- Quote actual function names, variable names, and values from the source files
- If a guard/check is missing, name the exact location where it belongs
- If logic is wrong, quote the wrong expression AND what it should be instead
- Keep each field to 1–2 sentences — be specific, not vague

Return ONLY a valid JSON array — no markdown fences, no preamble, no explanation."""

    user = (
        f"## Bug Report\n{problem_statement[:3500]}\n\n"
        f"## Relevant Source Files\n{file_ctx}\n\n"
        f"Generate {n} test failure hypotheses as a JSON array:"
    )

    raw = llm.complete(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
        temperature=0.25,
        max_tokens=1200,
    )
    if not raw:
        logger.warning("Hypothesis synthesis: LLM returned empty")
        return []

    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # Strip markdown fences if present
    if "```" in raw:
        m = re.search(r'```(?:json)?\n?(.*?)```', raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()

    # Try to parse the full response, then fall back to first JSON array found
    for candidate in [raw, re.search(r'\[.*\]', raw, re.DOTALL)]:
        text = candidate if isinstance(candidate, str) else (
            candidate.group(0) if candidate else ""
        )
        if not text:
            continue
        try:
            hyps = json.loads(text)
            if isinstance(hyps, list):
                valid = [h for h in hyps if isinstance(h, dict) and "failure_mode" in h]
                if valid:
                    valid.sort(key=lambda h: h.get("confidence", 0.0), reverse=True)
                    logger.info(
                        "Hypotheses generated: %d (top confidence=%.2f — %s)",
                        len(valid),
                        valid[0].get("confidence", 0.0),
                        valid[0].get("failure_mode", "")[:80],
                    )
                    return valid[:n]
        except (json.JSONDecodeError, ValueError):
            continue

    logger.warning("Hypothesis synthesis parse failed: %s", raw[:200])
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
    """
    Score a patch candidate without executing it.

    Weights:
      35% format      — is it a valid unified diff?
      35% relevance   — does it address concepts from the bug report?
      30% completeness — is it substantive? (penalises tiny stubs)
    """

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
        """
        Score based on the number and substance of added lines.
        Penalises stubs (< 100 chars or < 3 added lines).
        Rewards patches that have real content (100–5000 chars).
        """
        lines  = patch.strip().splitlines()
        n_plus = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
        if n_plus == 0:
            return 0.0
        # Graduated scoring — tiny patches get low scores
        patch_len = len(patch)
        if patch_len < 100 or n_plus < 3:
            return 0.15
        if patch_len < 300:
            return 0.50
        if patch_len < 800:
            return 0.75
        return 1.0


# ==============================================================================
# PLT SELF-CONSISTENCY CHECKER  ← NEW
# ==============================================================================

def llm_plt_consistency_check(
    candidates: list[tuple[str, float]],   # (patch, heuristic_score)
    hypotheses: list[dict],
    task: "SWETask",
    llm: "LLMClient",
) -> list[tuple[str, float]]:
    """
    PLT (Programming Logic & Technique) self-consistency check.

    Given N patch candidates and the inferred test failure hypotheses,
    make ONE LLM call that asks:
      "For each patch, does its logic actually fix the root causes described?
       Does the technique address the expected→actual gap in each hypothesis?"

    Returns the same candidates list with scores blended:
      final_score = 0.40 * heuristic_score + 0.60 * plt_score

    Falls back to original heuristic scores if the LLM call fails.
    Only runs when we have ≥1 hypothesis with confidence ≥ 0.4.
    """
    # Skip if no useful hypotheses — PLT check needs something to check against
    useful_hyps = [h for h in hypotheses if h.get("confidence", 0) >= 0.4]
    if not useful_hyps:
        logger.info("PLT check skipped: no high-confidence hypotheses")
        return candidates

    # Skip if all patches are empty/trivial
    non_empty = [(p, s) for p, s in candidates if len(p.strip()) > 50]
    if not non_empty:
        logger.info("PLT check skipped: all patches trivial")
        return candidates

    # Build compact patch summaries (truncated for token budget)
    patch_blocks = []
    for i, (patch, _) in enumerate(candidates, 1):
        # Show first 600 chars of each patch — enough to see the logic
        snippet = patch[:600] + (" …[truncated]" if len(patch) > 600 else "")
        patch_blocks.append(f"=== PATCH {i} ===\n{snippet}")
    patches_text = "\n\n".join(patch_blocks)

    # Summarise hypotheses concisely
    hyp_lines = []
    for i, h in enumerate(useful_hyps[:2], 1):   # top 2 only for brevity
        hyp_lines.append(
            f"H{i} (conf={h.get('confidence', 0):.0%}): "
            f"root_cause={h.get('root_cause', '?')} | "
            f"fix_needed={h.get('fix_hint', '?')}"
        )
    hyps_text = "\n".join(hyp_lines)

    system = """\
You are a senior code reviewer performing a Programming Logic & Technique (PLT) check.
You are given N patch candidates for a bug fix, plus inferred root-cause hypotheses.

For each patch, evaluate:
1. Does the patch's added/removed code DIRECTLY address the root_cause in the hypotheses?
2. Is the programming technique correct? (no off-by-one, no wrong operator, no missing guard)
3. Would the fix_hint in the hypothesis be satisfied by this patch's logic?

Return a JSON object with key "rankings" — an array of N objects in the SAME ORDER as
the patches, each with:
  "patch_index": int (1-based)
  "plt_score":   float 0.0–1.0 (1.0 = logic is perfectly sound and matches hypothesis)
  "reason":      string (one short sentence explaining the score)

Return ONLY valid JSON — no markdown, no preamble."""

    user = (
        f"## Bug: {task.problem_statement[:800]}\n\n"
        f"## Root-Cause Hypotheses\n{hyps_text}\n\n"
        f"## Patch Candidates\n{patches_text}\n\n"
        "Score each patch's PLT validity as JSON:"
    )

    raw = llm.complete(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
        temperature=0.1,
        max_tokens=512,
    )

    if not raw:
        logger.warning("PLT check: LLM returned empty — keeping heuristic scores")
        return candidates

    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    if "```" in raw:
        m = re.search(r'```(?:json)?\n?(.*?)```', raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()

    try:
        data     = json.loads(raw)
        rankings = data.get("rankings", [])
        if not isinstance(rankings, list) or len(rankings) != len(candidates):
            raise ValueError(f"Expected {len(candidates)} rankings, got {len(rankings)}")

        blended = []
        for i, (patch, h_score) in enumerate(candidates):
            entry = next(
                (r for r in rankings if r.get("patch_index") == i + 1),
                rankings[i] if i < len(rankings) else {}
            )
            plt_score = float(entry.get("plt_score", 0.5))
            plt_score = max(0.0, min(1.0, plt_score))   # clamp
            reason    = entry.get("reason", "")

            final = 0.40 * h_score + 0.60 * plt_score
            blended.append((patch, final))
            logger.info(
                "PLT branch %d/%d: heuristic=%.3f plt=%.3f final=%.3f — %s",
                i + 1, len(candidates), h_score, plt_score, final, reason[:80]
            )

        return blended

    except Exception as e:
        logger.warning("PLT check parse error (%s) — keeping heuristic scores", e)
        return candidates


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
                message = resp.json()["choices"][0]["message"]
                content = message.get("content") or message.get("reasoning_content", "")
                if not content:
                    logger.error("LLM returned content=null and reasoning_content=null")
                    return ""
                logger.info("LLM returned %d chars%s", len(content),
                            " (from reasoning_content)" if not message.get("content") else "")
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
    hypotheses: list[dict] | None = None,   # Stage 1.5 output
    prior_patch: str | None = None,
    iteration: int = 0, 
) -> list[dict]:
    """
    Build the final patch generation prompt.
    Includes hypotheses (if available) as the primary signal for what to fix,
    plus file context and optional exploration history.
    """
    # ── Format exploration history ──────────────────────────────────────────
    exploration_log = ""
    if history:
        parts = [obs.render() for obs in history]
        exploration_log = (
            "\n\n## Exploration History (bash commands + real output)\n\n"
            + "\n\n".join(parts)
        )

    # ── Format fetched files (up to 200 lines each — was 120) ──────────────
    file_context = ""
    if fetched_files:
        sections = []
        for fp, content in fetched_files.items():
            lines = content.splitlines()
            if len(lines) > 200:
                content = "\n".join(lines[:200]) + f"\n[...{len(lines)-200} more lines]"
            sections.append(f"### {fp}\n```\n{content}\n```")
        file_context = "\n\n## Source Files\n\n" + "\n\n".join(sections)

    # ── Format located files hint ───────────────────────────────────────────
    located_hint = ""
    if located_paths:
        located_hint = (
            "\n\nFiles identified for modification:\n"
            + "\n".join(f"  - {p}" for p in located_paths)
        )

    # ── Format hypothesis block (Stage 1.5) ────────────────────────────────
    hypothesis_block = ""
    if hypotheses:
        lines = [
            "\n\n## Inferred Test Failure Modes",
            "(Synthesised from the bug report + source files — use these to guide your patch)\n",
        ]
        for i, h in enumerate(hypotheses, 1):
            conf        = h.get("confidence", 0.0)
            failure     = h.get("failure_mode", "")
            expected    = h.get("expected_value", "")
            actual      = h.get("actual_value", "")
            root_cause  = h.get("root_cause", "")
            fix_hint    = h.get("fix_hint", "")
            lines.append(f"### Hypothesis {i}  (confidence {conf:.0%})")
            if failure:    lines.append(f"**Failing assertion:** {failure}")
            if expected:   lines.append(f"**Test expects:**      {expected}")
            if actual:     lines.append(f"**Bug produces:**      {actual}")
            if root_cause: lines.append(f"**Root cause:**        {root_cause}")
            if fix_hint:   lines.append(f"**Required fix:**      {fix_hint}")
            lines.append("")
        hypothesis_block = "\n" + "\n".join(lines)

    # ── System prompt — stronger when we have hypotheses ───────────────────
    if hypotheses:
        system = (
            "You are an expert software engineer. Your task is to produce a unified diff "
            "patch that fixes the described bug.\n\n"
            "You have been given INFERRED TEST FAILURE HYPOTHESES based on the bug report "
            "and source code analysis. These hypotheses describe what automated tests are "
            "currently failing and why. Your patch MUST satisfy the hypotheses: the fix "
            "should make the expected values match the actual values by correcting the "
            "identified root cause.\n\n"
            "OUTPUT FORMAT: unified diff ONLY, starting with diff --git\n"
            "- Copy context lines CHARACTER-FOR-CHARACTER from the source files shown\n"
            "- Include exactly 3 unchanged context lines before/after each change\n"
            "- Patch may span multiple files\n"
            "- No <think> tags, no markdown fences, no explanation\n"
        )
    else:
        system = (
            "You are an expert software engineer. Based on your exploration, "
            "generate the exact patch to fix the described bug.\n\n"
            "OUTPUT FORMAT: unified diff ONLY, starting with diff --git\n"
            "- Copy context lines CHARACTER-FOR-CHARACTER from what you saw\n"
            "- Include exactly 3 unchanged context lines before/after changes\n"
            "- May span multiple files\n"
            "- No <think> tags, no markdown fences, no explanation\n"
        )

    # ── Prior patch block (iteration > 0 only) ─────────────────────────
    prior_patch_block = ""
    if prior_patch and iteration > 0:
        snippet = prior_patch[:2000] + (" …[truncated]" if len(prior_patch) > 2000 else "")
        prior_patch_block = (
            "\n\n## Previous Best Patch (improve on this)\n\n"
            f"```diff\n{snippet}\n```\n\n"
            "The above patch was the best attempt so far but may still be "
            "incomplete or logically wrong. Study it carefully:\n"
            "- What did it get right?\n"
            "- What is missing or incorrect?\n"
            "- How can you fix its weaknesses while preserving its strengths?\n"
        )

    # ── User message — hypotheses placed prominently before files ───────────
    closing = (
        "\n\nNow output the IMPROVED unified diff patch:"
        if prior_patch_block else
        "\n\nNow output the final unified diff patch:"
    )
    
    user = (
        f"Repository: {task.repo}  commit: {task.base_commit or 'HEAD'}\n\n"
        f"## Bug Report\n\n{task.problem_statement[:4000]}"   # was :2000
        f"{hypothesis_block}"                                  # ← Stage 1.5 output
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
        action = "patch" if ("diff --git" in content or "--- a/" in content) else "bash"

    return action, content


# ==============================================================================
# PURPLE AGENT
# ==============================================================================

class PurpleAgent:
    """
    4-stage pipeline:
      Stage 1:   LLM localization (GitHub tree → LLM → file paths → fetch files)
      Stage 1.5: Synthetic test failure hypothesis generation         ← NEW
      Stage 2:   MCTS repair (parallel patch generation with PRM selection)
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
            "id":            session_id,
            "task":          task,
            "history":       [],          # list[Observation]
            "last_command":  "",          # last bash command sent
            "located_paths": None,        # set after Stage 1
            "fetched_files": {},          # static files from GitHub
            "hypotheses":    None,        # set after Stage 1.5  ← NEW
            "mcts":          MCTSEngine(branches=MCTS_BRANCHES),
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
        """
        Pipeline:
          1. Localize (first call only) → fetch files
          1.5. Synthesise test failure hypotheses
          2. MCTS repair → return patch
        """
        task = session["task"]
        logger.info("[%s] Running pipeline", session["id"][:20])

        # Stage 1 + 1.5 on first turn
        if session["located_paths"] is None:
            await self._run_localization(session)

        logger.info("[%s] Running MCTS repair", session["id"][:20])
        patch = await self._mcts_repair(session)
        return {"action": "patch", "content": patch}

    # ── Stage 1 + 1.5: Localization & Hypothesis Generation ──────────────────

    async def _run_localization(self, session: dict):
        """
        Stage 1:   GitHub tree → LLM localization → PARALLEL file fetch (up to 5).
        Stage 1.5: Hypothesis synthesis with hard 20s timeout (degrades to []).

        Timing profile for stages 1+1.5 (no per-task timeout on AgentBeats):
          tree fetch   :  ~1s
          localization :  ~3s
          file fetches :  ~2s   ← asyncio.gather (was sequential ~8s)
          hypotheses   :  ~8s   (LLM timeout=90s per call in LLMClient)
          ─────────────────────
          typical      : ~14s   (vs. ~8s in v3, well within 300 min/shard)
        """
        task = session["task"]
        sid  = session["id"][:20]

        # ── Stage 1a: repo tree ──────────────────────────────────────────────
        tree = await asyncio.to_thread(
            get_repo_tree, task.repo, task.base_commit or "HEAD"
        )

        if not tree:
            session["located_paths"] = []
            session["hypotheses"]    = []
            logger.warning("[%s] Tree fetch failed", sid)
            return

        # ── Stage 1b: LLM localization ───────────────────────────────────────
        located = await asyncio.to_thread(
            llm_localize,
            task.problem_statement,
            task.repo,
            tree,
            self.llm,
        )
        session["located_paths"] = located

        # ── Stage 1c: parallel file fetch ────────────────────────────────────
        # All files fetched simultaneously — saves ~6s vs. sequential loop
        ref = task.base_commit or "HEAD"
        fetch_coros = [
            asyncio.to_thread(fetch_file_raw, task.repo, ref, fp)
            for fp in located[:5]
        ]
        results = await asyncio.gather(*fetch_coros, return_exceptions=True)
        for fp, content in zip(located[:5], results):
            if isinstance(content, Exception):
                logger.warning("[%s] Fetch error %s: %s", sid, fp, content)
            elif content:
                session["fetched_files"][fp] = content
                logger.info("Fetched: %s (%d chars)", fp, len(content))

        # ── Stage 1.5: hypothesis synthesis with hard timeout ────────────────
        if not session["fetched_files"]:
            session["hypotheses"] = []
            logger.warning("[%s] No files fetched — hypothesis synthesis skipped", sid)
            return

        logger.info("[%s] Synthesising test failure hypotheses …", sid)
        hypotheses = await asyncio.to_thread(
            llm_synthesize_hypotheses,
            task.problem_statement,
            session["fetched_files"],
            self.llm,
        )
        session["hypotheses"] = hypotheses
        logger.info("[%s] Hypotheses ready: %d", sid, len(hypotheses))

    # ── Stage 2: MCTS Repair ──────────────────────────────────────────────────
    async def _mcts_repair(self, session):
        task = session["task"]
        hypotheses = session.get("hypotheses") or []
    
        best_patch = ""
        best_score = 0.0
    
        for iteration in range(MCTS_ITERATIONS):
            # Build prompt — on iteration>0, include the best patch so far
            # so the model can see what to improve
            msgs = _build_repair_messages(
                task,
                session["history"],
                session["located_paths"] or [],
                session["fetched_files"],
                hypotheses,
                prior_patch=best_patch if iteration > 0 else None,  # ← KEY
                iteration=iteration,
            )
        
            temps = [0.15 + (0.65 / max(MCTS_BRANCHES - 1, 1)) * i
                     for i in range(MCTS_BRANCHES)]
        
            raws = await asyncio.gather(*[
                asyncio.to_thread(self.llm.complete, msgs, t, 2048)
                for t in temps
            ])
        
            candidates = []
            for i, raw in enumerate(raws):
                patch = _extract_patch(raw)
                score = self.prm.score(patch, task)
                candidates.append((patch, score))
                logger.info("[%s] Iter %d Branch %d/%d T=%.2f score=%.3f len=%d",
                            session["id"][:20], iteration+1, i+1,
                            MCTS_BRANCHES, temps[i], score, len(patch))
        
            # PLT check
            if hypotheses:
                candidates = await asyncio.to_thread(
                    llm_plt_consistency_check, candidates, hypotheses, task, self.llm
                )
        
            # MCTS select — NOW UCT is meaningful because root accumulates visits
            round_best = session["mcts"].select(candidates)
            round_score = next(s for p, s in candidates if p == round_best)
            session["mcts"].backpropagate(round_score)
        
            if round_score > best_score:
                best_score = round_score
                best_patch = round_best
                logger.info("[%s] Iter %d improved: score=%.3f",
                            session["id"][:20], iteration+1, best_score)
            else:
                logger.info("[%s] Iter %d no improvement — stopping early",
                            session["id"][:20], iteration+1)
                break   # early stopping if score didn't improve
    
        return best_patch


# ==============================================================================
# FASTAPI APP
# ==============================================================================

app   = FastAPI(title="Purple Coding Agent")
agent = PurpleAgent()

AGENT_CARD = {
    "name": "Purple Coding Agent",
    "description": (
        "SWE-bench agent: LLM localization → synthetic hypothesis synthesis → "
        "MCTS repair. v4 with Stage 1.5 test failure inference."
    ),
    "url": f"http://localhost:{PORT}/",
    "version": "4.1.0",
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
                "Localise the bug, infer failing test hypotheses, then generate "
                "a unified diff patch fixing the described issue."
            ),
            "tags": ["coding", "swe-bench", "patch", "hypothesis"],
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
    return {"status": "ok", "agent": "purple-coding-agent", "version": "4.1.0"}


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

    task_data["session_id"] = context_id

    ps = task_data.get("problem_statement", "")
    logger.info("context_id=%s  ps_len=%d  repo=%s  commit=%s",
                context_id[:20], len(ps),
                task_data.get("repo", "?"),
                (task_data.get("base_commit", "") or "")[:12] or "HEAD")

    loop   = asyncio.get_event_loop()
    action = await loop.run_in_executor(None, agent.respond, task_data)

    if action.get("action") == "patch":
        artifact_text = action.get("content", "")
    else:
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