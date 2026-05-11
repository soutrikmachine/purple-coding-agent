"""
HypothesisGenerator — GSRM (Group Sampling Reward Mechanism)

Design:
  Stage 1 — GROUP SAMPLING
    Generate g_size candidate hypotheses at temperature=0.7 (diverse).
    Each candidate: title, file_lead, verify_cmd, reasoning.
    use_thinking=False — structured JSON output, fast response needed.

  Stage 2 — EXECUTION REWARD SCORING
    Execute each verify_cmd in the real container.
    Score from actual bash output — grounded reality, not guessing.

    Reward signals:
      +3  output contains failure patterns (FAILED/Error/Traceback/assert)
          → bug actively reproduced
      +2  non-zero exit with output → found something, on the right track
      +1  zero exit with meaningful output → confirmed file/function exists
      +1  output has keyword overlap with problem_statement (≥3 keywords)
       0  empty output → command missed, wrong path

  Stage 3 — RANKING
    Sort by score descending. Best hypothesis passed to REPL pre-loop
    as verify_cmd — agent enters turn 1 already seeing real test output.
"""

import json
import logging
import re
from typing import List, Dict, Any

from ..core.llm_client import LLMClient

logger = logging.getLogger(__name__)


class HypothesisGenerator:

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    # ── Stage 1: Group Sampling ────────────────────────────────────────────────

    async def generate_group(
        self,
        problem_statement: str,
        repo_skeleton: str,
        g_size: int = 3,
        hints_text: str = "",
        docker=None,          # DockerBridge — for execution reward scoring
        repo_dir: str = "/",  # repo root inside container
    ) -> List[Dict[str, Any]]:
        """
        Generate candidates via group sampling then score by real execution.
        Returns candidates sorted by reward score (best first).
        """
        system_prompt = (
            "You are a diagnostic engine for a stateful engineering agent.\n"
            "Given a bug report and a repository file tree, identify likely failure "
            "points and provide actionable bash commands to verify each one.\n\n"
            "Return a JSON array containing EXACTLY your hypotheses.\n"
            "Format each object strictly as:\n"
            "{\n"
            '  "title":      "Short description of the suspected root cause",\n'
            '  "file_lead":  "path/to/suspect/file (relative to repo root)",\n'
            '  "verify_cmd": "exact bash command to confirm the bug",\n'
            '  "reasoning":  "Why this is the likely cause"\n'
            "}\n\n"
            "Rules for verify_cmd:\n"
            "- Single runnable bash command\n"
            "- Prefer: pytest tests/specific_test.py -x  or  grep -n 'pattern' file\n"
            "- Should REPRODUCE or LOCATE the bug, not fix it\n"
            "- Use relative paths from the repo root"
        )

        hints_block = ""
        if hints_text and hints_text.strip():
            hints_block = (
                f"\n## Additional Hints (from benchmark annotators)\n"
                f"{hints_text.strip()}\n"
            )
            logger.info("hints_text injected (%d chars)", len(hints_text))

        user_prompt = (
            f"## Bug Report\n{problem_statement}\n"
            f"{hints_block}\n"
            f"## Repository File Tree\n{repo_skeleton[:15000]}\n\n"
            f"Generate {g_size} distinct debugging hypotheses as a JSON array:"
        )

        # use_thinking=False — structured JSON output task.
        # Thinking adds 15-20s latency and causes the 15s timeout every time.
        raw_response = await self.llm.generate_step(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.7,
            use_thinking=False,
        )

        candidates = self._parse_hypotheses(raw_response)

        if not candidates:
            logger.warning("No hypotheses parsed — returning empty")
            return []

        logger.info("Group sampling: %d candidates generated", len(candidates))

        # ── Stage 2: Execution Reward Scoring ─────────────────────────────────
        if docker is not None:
            candidates = await self._score_by_execution(
                candidates, docker, repo_dir, problem_statement
            )
        else:
            logger.warning("No DockerBridge — skipping execution scoring")
            for h in candidates:
                h["reward_score"] = 0
                h["exec_output"]  = ""

        return candidates

    # ── Stage 2: Execution Reward Scoring ─────────────────────────────────────

    async def _score_by_execution(
        self,
        candidates: List[Dict[str, Any]],
        docker,
        repo_dir: str,
        problem_statement: str,
    ) -> List[Dict[str, Any]]:
        """
        Execute each verify_cmd and score by real output.
        """
        import asyncio

        ps_keywords = set(re.findall(r'\b[a-zA-Z_]\w{3,}\b', problem_statement.lower()))

        failure_patterns = re.compile(
            r'FAILED|Error|Traceback|assert|FAIL:|panic:|exception|undefined|'
            r'TypeError|ValueError|AttributeError|ImportError|ModuleNotFoundError|'
            r'nil pointer|index out of range|no such file',
            re.IGNORECASE,
        )

        for i, hyp in enumerate(candidates):
            verify_cmd = hyp.get("verify_cmd", "").strip()
            score      = 0
            exec_output = ""

            if not verify_cmd:
                hyp["reward_score"] = 0
                hyp["exec_output"]  = ""
                continue

            try:
                ec, output = await asyncio.to_thread(
                    docker.execute_command,
                    f"cd {repo_dir} && {verify_cmd}",
                    10,  # 10s cap per verify_cmd
                )
                exec_output = output.strip()

                if exec_output:
                    if failure_patterns.search(exec_output):
                        score += 3   # bug reproduced — highest value
                    elif ec != 0:
                        score += 2   # non-zero exit with output
                    elif len(exec_output) >= 10:
                        score += 1   # confirmed something exists

                    # semantic overlap bonus
                    out_words = set(re.findall(r'\b[a-zA-Z_]\w{3,}\b', exec_output.lower()))
                    if len(ps_keywords & out_words) >= 3:
                        score += 1

                logger.info(
                    "Hypothesis %d: score=%d exit=%d out=%d chars cmd=%s",
                    i + 1, score, ec, len(exec_output), verify_cmd[:60],
                )

            except Exception as e:
                logger.warning("Hypothesis %d exec error: %s", i + 1, str(e)[:100])
                exec_output = f"[exec error: {e}]"

            hyp["reward_score"] = score
            hyp["exec_output"]  = exec_output[:500]

        candidates.sort(key=lambda h: h.get("reward_score", 0), reverse=True)

        logger.info(
            "GSRM ranking done. Best: score=%d title=%s",
            candidates[0].get("reward_score", 0),
            candidates[0].get("title", "?")[:60],
        )
        return candidates

    # ── Parsing ────────────────────────────────────────────────────────────────

    def _parse_hypotheses(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        try:
            text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
            text = re.sub(r'```(?:json)?\s*\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
            text = text.strip()

            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    valid = [h for h in parsed if isinstance(h, dict) and "file_lead" in h]
                    if valid:
                        return valid
            except json.JSONDecodeError:
                pass

            match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return [h for h in parsed if isinstance(h, dict) and "file_lead" in h]

        except Exception as e:
            logger.error("Parse failed: %s | raw: %s", e, text[:200])

        return []

    # ── Primer formatting ──────────────────────────────────────────────────────

    def format_for_primer(self, hypotheses: List[Dict[str, Any]]) -> str:
        if not hypotheses:
            return (
                "## Initial Diagnostic Hypotheses\n"
                "No hypotheses generated. Proceed with manual exploration."
            )

        primer = [
            "## Initial Diagnostic Hypotheses",
            "(Ranked by execution reward score — try Strategy 1 first)\n",
        ]
        for i, h in enumerate(hypotheses, 1):
            score = h.get("reward_score", 0)
            out   = h.get("exec_output", "")
            primer.append(
                f"### Strategy {i}: {h.get('title', 'Unknown')} [reward_score={score}]"
            )
            primer.append(f"- **Root Cause Lead:** `{h.get('file_lead', '')}`")
            primer.append(f"- **Verification Command:** `{h.get('verify_cmd', '')}`")
            primer.append(f"- **Reasoning:** {h.get('reasoning', '')}")
            if out:
                primer.append(f"- **Execution Output:**\n```\n{out[:300]}\n```")
            primer.append("")

        return "\n".join(primer)