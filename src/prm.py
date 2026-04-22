"""
Programmable Process Reward Model (PRM)
========================================
Scores candidate actions and observations to guide MCTS.

Three reward layers:

  Hard Logic  – deterministic rules (valid bash syntax, diff format, etc.)
  Env Logic   – observation-based signals (stdout mentions test pass/fail)
  Soft Logic  – heuristics (TIR quality, file discovery, specificity)

Total reward ∈ [0, 1].
"""

from __future__ import annotations

import re
import subprocess
import logging

logger = logging.getLogger("purple_agent.prm")


class ProgrammablePRM:
    """Score actions and observations for MCTS guidance."""

    # Weights for reward components
    W_FORMAT = 0.20
    W_RELEVANCE = 0.35
    W_EXECUTION = 0.45

    # ── Static Scoring (before execution) ─────────────────────────────────────

    def score_static(self, action: dict, task) -> float:
        """
        Score a candidate action using only static analysis.
        Used by MCTS to rank branches before any execution.
        """
        a_type = action.get("action", "bash")
        content = action.get("content", "")

        score = 0.0

        # Format reward: is the action well-formed?
        score += self.W_FORMAT * self._format_reward(a_type, content)

        # Relevance reward: does content mention relevant terms?
        score += self.W_RELEVANCE * self._relevance_reward(content, task)

        return min(score, 1.0)

    # ── Observation Scoring (after execution) ─────────────────────────────────

    def score_observation(self, obs: dict, task) -> float:
        """
        Score the result of an executed action.
        Called by MCTS backpropagation after green agent returns an observation.
        """
        stdout = obs.get("stdout", "")
        stderr = obs.get("stderr", "")

        score = 0.0

        # Execution reward: did tests pass? Did we find the bug?
        score += self.W_EXECUTION * self._execution_reward(stdout, stderr, task)

        # Bonus for discovering file locations
        score += 0.15 * self._discovery_reward(stdout, task)

        # Penalty for errors / empty output
        if stderr and not stdout:
            score -= 0.10

        return max(0.0, min(score + 0.3, 1.0))  # baseline 0.3

    # ── Reward Components ─────────────────────────────────────────────────────

    def _format_reward(self, action_type: str, content: str) -> float:
        """Check that the action is syntactically valid."""
        if action_type == "bash":
            # Should look like a shell command, not empty
            if not content.strip():
                return 0.0
            if len(content) > 500:  # suspiciously long bash one-liner
                return 0.5
            return 1.0

        if action_type == "debug":
            if not content.strip():
                return 0.0
            return 0.9

        if action_type == "patch":
            # Must look like a unified diff
            has_header = "diff --git" in content or "--- a/" in content
            has_hunks = "@@" in content
            has_changes = "+\n" in content or "-\n" in content or "+" in content
            if has_header and has_hunks and has_changes:
                return 1.0
            if has_header or has_hunks:
                return 0.5
            return 0.1

        return 0.5

    def _relevance_reward(self, content: str, task) -> float:
        """
        Check if the action mentions terms related to the bug.
        Higher score if it targets specific files/functions from the problem.
        """
        if not task.problem_statement:
            return 0.5

        # Extract candidate tokens from problem statement
        ps_tokens = set(re.findall(r"\b\w{4,}\b", task.problem_statement.lower()))
        content_tokens = set(re.findall(r"\b\w{4,}\b", content.lower()))

        if not ps_tokens:
            return 0.5

        overlap = len(ps_tokens & content_tokens) / len(ps_tokens)
        return min(overlap * 2, 1.0)  # scale up

    def _execution_reward(self, stdout: str, stderr: str, task) -> float:
        """
        Parse test results and error patterns from stdout/stderr.
        """
        stdout_lower = stdout.lower()
        stderr_lower = stderr.lower()

        # Strong positive: pytest reports
        if "passed" in stdout_lower and "failed" not in stdout_lower:
            return 1.0
        if re.search(r"\d+ passed", stdout_lower):
            return 0.9
        if "error" not in stderr_lower and "traceback" not in stderr_lower:
            if stdout.strip():  # got output, no errors
                return 0.6

        # Moderate positive: found the file / relevant code
        if any(f in stdout for f in task.fail_to_pass):
            return 0.8

        # Negative: test failure or import error
        if "failed" in stdout_lower or "error" in stderr_lower:
            return 0.2

        return 0.4  # neutral

    def _discovery_reward(self, stdout: str, task) -> float:
        """Bonus if stdout contains file paths (agent found relevant code)."""
        file_patterns = re.findall(r'[\w/]+\.py(?::\d+)?', stdout)
        if not file_patterns:
            return 0.0

        # Extra bonus if the file is mentioned in the problem statement
        ps_lower = task.problem_statement.lower()
        for fp in file_patterns:
            base = fp.split("/")[-1].replace(".py", "")
            if base in ps_lower:
                return 1.0

        return 0.5 if file_patterns else 0.0
