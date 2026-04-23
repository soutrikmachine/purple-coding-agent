"""
PurpleAgent — Fixed for SWE-bench Pro
======================================
Key fixes vs original:
  1. HF_TOKEN passed through to LLMClient for HuggingFace inference API
  2. SWE-bench message format: instance_id, base_commit, repo fields handled
  3. Multi-turn conversation properly maintained per instance_id
  4. Fallback: if LLM fails, return empty patch rather than crashing
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


class PurpleAgent:
    def __init__(
        self,
        model_base_url: str = "https://api-inference.huggingface.co/v1",
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        hf_token: str = "",
        max_turns: int = 15,
        mcts_branches: int = 3,
        temperature: float = 0.6,
        use_mcts: bool = True,
    ):
        self.llm = LLMClient(
            base_url=model_base_url,
            model=model_name,
            api_key=hf_token,  # FIX: pass token so HF API calls succeed
        )
        self.prm = ProgrammablePRM()
        self.max_turns = max_turns
        self.mcts_branches = mcts_branches
        self.temperature = temperature
        self.use_mcts = use_mcts
        self._sessions: dict[str, dict[str, Any]] = {}

    def respond(self, message: dict) -> dict:
        # FIX: SWE-bench sends instance_id — use it as stable session key
        session_id = (
            message.get("instance_id")
            or message.get("session_id")
            or self._infer_session_id(message)
        )

        if session_id not in self._sessions:
            session = self._init_session(session_id, message)
        else:
            session = self._sessions[session_id]

        try:
            return self._step(session, message)
        except Exception as e:
            logger.exception("[%s] Agent step failed: %s", session_id, e)
            return {"action": "patch", "content": ""}  # safe fallback

    def _init_session(self, session_id: str, message: dict) -> dict:
        task = SWETask(
            # FIX: SWE-bench green agent sends these fields
            problem_statement=message.get("problem_statement", ""),
            cwd=message.get("cwd", "/workspace/repo"),
            hints_text=message.get("hints_text", ""),
            python_version=message.get("python_version", "3.9"),
            fail_to_pass=message.get("fail_to_pass", []) or [],
            pass_to_pass=message.get("pass_to_pass", []) or [],
            repo=message.get("repo", ""),
            instance_id=message.get("instance_id", ""),
            base_commit=message.get("base_commit", ""),
        )

        state_mgr = NodeStateManager()
        root = MCTSNode(state=state_mgr.new_root_state(task))

        session = {
            "id": session_id,
            "task": task,
            "state_mgr": state_mgr,
            "mcts": MCTSEngine(root, branches=self.mcts_branches),
            "turn": 0,
            "history": [],
            "patch_attempts": 0,
            "submitted_patch": None,
        }
        self._sessions[session_id] = session
        logger.info("[%s] Session init: repo=%s instance=%s", session_id, task.repo, task.instance_id)
        return session

    def _step(self, session: dict, message: dict) -> dict:
        task: SWETask = session["task"]
        session["turn"] += 1
        turn = session["turn"]

        # Record observation from previous action
        if "stdout" in message or "stderr" in message:
            obs = {
                "cwd": message.get("cwd", task.cwd),
                "stdout": message.get("stdout", ""),
                "stderr": message.get("stderr", ""),
            }
            if session["history"]:
                session["history"][-1]["observation"] = obs
                score = self.prm.score_observation(obs, task)
                session["mcts"].backpropagate(score)

        # Force patch when approaching turn limit
        if turn >= self.max_turns - 2 and session["submitted_patch"] is None:
            logger.info("[%s] Near turn limit — forcing patch", session["id"])
            return self._force_patch(session)

        action = self._mcts_action(session) if (self.use_mcts and turn > 1) else self._greedy_action(session)
        session["history"].append({"action": action})
        logger.info("[%s] turn=%d action=%s", session["id"], turn, action.get("action"))
        return action

    def _greedy_action(self, session: dict) -> dict:
        messages = self._build_messages(session)
        raw = self.llm.complete(messages, temperature=self.temperature, max_tokens=1024)
        return self._parse_action(raw, session)

    def _mcts_action(self, session: dict) -> dict:
        messages = self._build_messages(session)
        candidates = []
        for _ in range(self.mcts_branches):
            raw = self.llm.complete(messages, temperature=self.temperature + 0.1, max_tokens=1024)
            action = self._parse_action(raw, session)
            score = self.prm.score_static(action, session["task"])
            candidates.append((action, score))
        best_action, _ = session["mcts"].select_action(candidates)
        return best_action

    def _force_patch(self, session: dict) -> dict:
        task = session["task"]
        history_text = self._format_history(session["history"])
        force_prompt = (
            f"Based on your exploration:\n{history_text}\n\n"
            f"Problem: {task.problem_statement[:500]}\n\n"
            "Produce the final unified diff patch now. Output ONLY the patch in action XML format."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": force_prompt},
        ]
        raw = self.llm.complete(messages, temperature=0.2, max_tokens=2048)
        action = self._parse_action(raw, session)
        if action.get("action") != "patch":
            content = action.get("content", "")
            if "diff --git" in content or "--- a/" in content:
                action = {"action": "patch", "content": content}
            else:
                action = {"action": "patch", "content": ""}
        session["submitted_patch"] = action["content"]
        return action

    def _build_messages(self, session: dict) -> list[dict]:
        task = session["task"]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": build_user_prompt(task)})
        for entry in session["history"]:
            action = entry.get("action", {})
            messages.append({"role": "assistant", "content": self._format_action_xml(action)})
            obs = entry.get("observation")
            if obs:
                messages.append({"role": "user", "content": build_observation_prompt(obs)})
        return messages

    def _parse_action(self, raw: str, session: dict) -> dict:
        raw = raw.strip()
        thought = self._extract_tag(raw, "thought") or ""
        command = (self._extract_tag(raw, "command") or "bash").strip().lower()
        content = self._extract_tag(raw, "content") or ""

        if thought:
            logger.debug("[%s] Thought: %s", session["id"], thought[:150])

        if command not in ("bash", "debug", "patch"):
            if "diff --git" in content or raw.startswith("--- "):
                command = "patch"
            else:
                command = "bash"

        content = content.strip()
        if command == "bash":
            content = self._sanitize_bash(content)

        return {"action": command, "content": content}

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str | None:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return m.group(1).strip() if m else None

    @staticmethod
    def _sanitize_bash(cmd: str) -> str:
        for dangerous in ["rm -rf", "rm -f", "> /", "dd if=", "mkfs"]:
            if dangerous in cmd:
                return f"echo '[BLOCKED]'"
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
            lines.append(f"Step {i}: [{act.get('action')}] {act.get('content', '')[:80]}")
            if obs:
                lines.append(f"  → {obs.get('stdout', '')[:150]}")
        return "\n".join(lines)

    @staticmethod
    def _infer_session_id(message: dict) -> str:
        ps = message.get("problem_statement", "")
        return str(abs(hash(ps[:100])))
