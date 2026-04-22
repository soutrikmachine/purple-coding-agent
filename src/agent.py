"""
PurpleAgent: Core Orchestrator
==============================
Receives a SWE-bench task (problem_statement, cwd, hints, fail_to_pass tests)
and drives an MCTS-guided loop:
  1. LLM proposes candidate actions (bash / debug / patch)
  2. Actions are scored by the Programmable PRM
  3. MCTS selects the best branch
  4. Loop continues until a valid patch is produced or max_turns reached
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from llm_client import LLMClient
from mcts import MCTSEngine, MCTSNode
from prm import ProgrammablePRM
from prompts import SYSTEM_PROMPT, build_user_prompt, build_observation_prompt
from state import NodeStateManager

logger = logging.getLogger("purple_agent.agent")


# ── Task Representation ───────────────────────────────────────────────────────

@dataclass
class SWETask:
    problem_statement: str
    cwd: str = "/workspace/repo"
    hints_text: str = ""
    python_version: str = "3.9"
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    repo: str = ""


# ── PurpleAgent ───────────────────────────────────────────────────────────────

class PurpleAgent:
    """
    Stateless per-call agent.  Each call to `respond()` handles exactly one
    turn of the A2A conversation (the green agent expects one action per call).

    Internally we maintain session state indexed by conversation_id so that
    multi-turn tasks work correctly.
    """

    def __init__(
        self,
        model_base_url: str = "https://api-inference.huggingface.co/v1/",
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        api_key: str = "", # Added API key parameter
        max_turns: int = 15,
        mcts_branches: int = 3,
        temperature: float = 0.6,
        use_mcts: bool = True,
    ):
        self.llm = LLMClient(base_url=model_base_url, model=model_name, api_key=api_key)
        self.prm = ProgrammablePRM()
        self.max_turns = max_turns
        self.mcts_branches = mcts_branches
        self.temperature = temperature
        self.use_mcts = use_mcts

        # Per-session state: session_id -> dict
        self._sessions: dict[str, dict[str, Any]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def respond(self, message: dict) -> dict:
        """
        Process one A2A message from the green agent.

        message shape (initial turn):
            {
              "cwd": "/workspace/repo",
              "problem_statement": "...",
              "hints_text": "...",
              "python_version": "3.9",
              "fail_to_pass": [...],
            }

        message shape (subsequent turns, observation):
            {
              "session_id": "...",
              "cwd": "/workspace/repo/src",
              "stdout": "...",
              "stderr": "...",
            }

        Returns one of:
            {"action": "bash",  "content": "<bash command>"}
            {"action": "debug", "content": "<bash command with writes>"}
            {"action": "patch", "content": "<unified diff>"}
        """
        session_id = message.get("session_id") or self._infer_session_id(message)

        if session_id not in self._sessions:
            session = self._init_session(session_id, message)
        else:
            session = self._sessions[session_id]

        return self._step(session, message)

    # ── Session Lifecycle ─────────────────────────────────────────────────────

    def _init_session(self, session_id: str, message: dict) -> dict:
        task = SWETask(
            problem_statement=message.get("problem_statement", ""),
            cwd=message.get("cwd", "/workspace/repo"),
            hints_text=message.get("hints_text", ""),
            python_version=message.get("python_version", "3.9"),
            fail_to_pass=message.get("fail_to_pass", []),
            pass_to_pass=message.get("pass_to_pass", []),
            repo=message.get("repo", ""),
        )

        state_mgr = NodeStateManager()
        root = MCTSNode(state=state_mgr.new_root_state(task))

        session = {
            "id": session_id,
            "task": task,
            "state_mgr": state_mgr,
            "mcts": MCTSEngine(root, branches=self.mcts_branches),
            "turn": 0,
            "history": [],          # list of (action, observation) dicts
            "patch_attempts": 0,
            "submitted_patch": None,
        }
        self._sessions[session_id] = session
        logger.info("[%s] Session initialised for repo=%s", session_id, task.repo)
        return session

    # ── Reasoning Step ────────────────────────────────────────────────────────

    def _step(self, session: dict, message: dict) -> dict:
        task: SWETask = session["task"]
        session["turn"] += 1
        turn = session["turn"]

        # Record observation from green agent (skip on first turn)
        if "stdout" in message or "stderr" in message:
            obs = {
                "cwd": message.get("cwd", task.cwd),
                "stdout": message.get("stdout", ""),
                "stderr": message.get("stderr", ""),
            }
            if session["history"]:
                session["history"][-1]["observation"] = obs
                # Update MCTS node with PRM score
                score = self.prm.score_observation(obs, task)
                session["mcts"].backpropagate(score)
                logger.info("[%s] turn=%d PRM score=%.3f", session["id"], turn, score)

        # Safety: if we've used most turns, force a patch attempt
        if turn >= self.max_turns - 2 and session["submitted_patch"] is None:
            logger.info("[%s] Nearing turn limit – forcing patch generation", session["id"])
            return self._force_patch(session)

        # Choose next action
        if self.use_mcts and turn > 1:
            action = self._mcts_action(session)
        else:
            action = self._greedy_action(session)

        session["history"].append({"action": action})
        logger.info("[%s] turn=%d action=%s", session["id"], turn, action.get("action"))
        return action

    # ── Action Selection ──────────────────────────────────────────────────────

    def _greedy_action(self, session: dict) -> dict:
        """Single-sample action: just ask the LLM for one action."""
        messages = self._build_messages(session)
        raw = self.llm.complete(messages, temperature=self.temperature, max_tokens=1024)
        return self._parse_action(raw, session)

    def _mcts_action(self, session: dict) -> dict:
        """
        MCTS-guided action selection:
        1. Sample N candidate actions from LLM
        2. Score each with the static PRM (no execution yet)
        3. Select via UCT
        """
        messages = self._build_messages(session)
        candidates = []

        for _ in range(self.mcts_branches):
            raw = self.llm.complete(
                messages,
                temperature=self.temperature + 0.1,  # slight diversity
                max_tokens=1024,
            )
            action = self._parse_action(raw, session)
            static_score = self.prm.score_static(action, session["task"])
            candidates.append((action, static_score))

        # Let MCTS engine pick the best branch
        best_action, _ = session["mcts"].select_action(candidates)
        return best_action

    def _force_patch(self, session: dict) -> dict:
        """Ask the LLM to produce a final patch based on everything discovered."""
        task = session["task"]
        history_text = self._format_history(session["history"])

        force_prompt = (
            f"Based on your exploration so far:\n{history_text}\n\n"
            f"Problem: {task.problem_statement}\n\n"
            "Now produce the final unified diff patch to fix this bug. "
            "Output ONLY the patch in action format."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": force_prompt},
        ]
        raw = self.llm.complete(messages, temperature=0.2, max_tokens=2048)
        action = self._parse_action(raw, session)

        # Ensure it's a patch action
        if action.get("action") != "patch":
            # Try to extract diff from content
            content = action.get("content", "")
            if "diff --git" in content or "--- a/" in content:
                action = {"action": "patch", "content": content}
            else:
                # Fallback: empty patch
                action = {"action": "patch", "content": ""}

        session["submitted_patch"] = action["content"]
        return action

    # ── Prompt Construction ───────────────────────────────────────────────────

    def _build_messages(self, session: dict) -> list[dict]:
        task = session["task"]
        history = session["history"]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # First user message: task description
        messages.append({
            "role": "user",
            "content": build_user_prompt(task),
        })

        # Interleave history as assistant/user turns
        for entry in history:
            action = entry.get("action", {})
            messages.append({
                "role": "assistant",
                "content": self._format_action_xml(action),
            })
            obs = entry.get("observation")
            if obs:
                messages.append({
                    "role": "user",
                    "content": build_observation_prompt(obs),
                })

        return messages

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_action(self, raw: str, session: dict) -> dict:
        """
        Parse LLM output into an A2A action dict.
        Expected XML format:
            <thought>...</thought>
            <command>bash|debug|patch</command>
            <content>...</content>
        """
        raw = raw.strip()

        # Try XML parsing
        thought = self._extract_tag(raw, "thought") or ""
        command = (self._extract_tag(raw, "command") or "bash").strip().lower()
        content = self._extract_tag(raw, "content") or ""

        if thought:
            logger.debug("[%s] Thought: %s", session["id"], thought[:200])

        # Validate command
        if command not in ("bash", "debug", "patch"):
            # Heuristic: if it looks like a diff, call it a patch
            if "diff --git" in content or raw.startswith("--- "):
                command = "patch"
            else:
                command = "bash"

        # Clean up content
        content = content.strip()

        # Safety: limit bash commands to read-only in bash mode
        if command == "bash":
            content = self._sanitize_bash(content)

        return {"action": command, "content": content}

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str | None:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else None

    @staticmethod
    def _sanitize_bash(cmd: str) -> str:
        """Strip accidental write ops from bash commands."""
        # Remove obvious destructive ops
        dangerous = ["rm -rf", "rm -f", "> /", "dd if=", "mkfs"]
        for d in dangerous:
            if d in cmd:
                cmd = f"echo '[BLOCKED: {d}]'"
                break
        return cmd

    @staticmethod
    def _format_action_xml(action: dict) -> str:
        a = action.get("action", "bash")
        c = action.get("content", "")
        return f"<command>{a}</command>\n<content>{c}</content>"

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        lines = []
        for i, entry in enumerate(history, 1):
            act = entry.get("action", {})
            obs = entry.get("observation", {})
            lines.append(f"Turn {i}: {act.get('action')} -> {act.get('content', '')[:80]}")
            if obs:
                out = obs.get("stdout", "")[:200]
                lines.append(f"  Result: {out}")
        return "\n".join(lines)

    @staticmethod
    def _infer_session_id(message: dict) -> str:
        """Derive a stable session ID from the problem statement hash."""
        ps = message.get("problem_statement", "")
        return str(abs(hash(ps[:100])))
