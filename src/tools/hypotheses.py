import json
import logging
import re
from typing import List, Dict, Any

from ..core.llm_client import LLMClient

logger = logging.getLogger(__name__)


class HypothesisGenerator:
    """
    Group-Sampling hypothesis generator (GSRM).

    Generates g_size competing debugging leads at temperature=0.7 for diversity.
    hints_text (from the green agent's instances.jsonl) is incorporated when
    non-empty — it often contains test file paths, stack traces, or function names
    that are more specific than the problem_statement alone.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def generate_group(
        self,
        problem_statement: str,
        repo_skeleton: str,
        g_size: int = 2,
        hints_text: str = "",        # ← new: from instances.jsonl
    ) -> List[Dict[str, Any]]:

        system_prompt = (
            "You are a diagnostic engine for a stateful software engineering agent.\n"
            "Given a bug report, optional hints, and a repository file tree, "
            "identify the most likely failure points.\n\n"
            "Focus on actionable leads — specific files and bash commands "
            "the agent can run immediately to verify each hypothesis.\n\n"
            "Return a JSON array of exactly your hypotheses. "
            "Each object must have EXACTLY these keys:\n"
            "{\n"
            "  \"title\":      \"Short description of the suspected bug\",\n"
            "  \"file_lead\":  \"path/to/most/suspect/file (relative to repo root)\",\n"
            "  \"verify_cmd\": \"exact bash command to run first to confirm the bug\",\n"
            "  \"reasoning\":  \"Why this is the likely root cause\"\n"
            "}\n\n"
            "Rules:\n"
            "- Return ONLY valid JSON — no markdown fences, no preamble\n"
            "- verify_cmd must be a single runnable bash command\n"
            "- If hints mention a specific test or file, use it in file_lead/verify_cmd\n"
            "- Prefer targeted pytest/go test commands over broad test suite runs"
        )

        # Build user prompt — inject hints_text prominently when available
        hints_block = ""
        if hints_text and hints_text.strip():
            hints_block = f"\n## Additional Hints (from benchmark annotators)\n{hints_text.strip()}\n"
            logger.info("hints_text provided (%d chars) — injecting into hypothesis prompt", len(hints_text))

        user_prompt = (
            f"## Bug Report\n{problem_statement}\n"
            f"{hints_block}\n"
            f"## Repository File Tree\n{repo_skeleton[:12000]}\n\n"
            f"Generate {g_size} distinct debugging hypotheses as a JSON array:"
        )

        raw_response = await self.llm.generate_step(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.7,   # higher temp for diverse group sampling
        )

        return self._parse_hypotheses(raw_response)

    def _parse_hypotheses(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        try:
            # Strip Gemini thinking tags
            text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
            # Strip markdown fences
            text = re.sub(r'```(?:json)?\s*\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
            text = text.strip()

            # Try direct parse first
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    valid = [h for h in parsed if isinstance(h, dict) and "file_lead" in h]
                    if valid:
                        logger.info("Parsed %d hypotheses", len(valid))
                        return valid
            except json.JSONDecodeError:
                pass

            # Fallback: find first JSON array in text
            match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    valid = [h for h in parsed if isinstance(h, dict) and "file_lead" in h]
                    logger.info("Parsed %d hypotheses (fallback regex)", len(valid))
                    return valid

        except Exception as e:
            logger.error("Failed to parse hypotheses: %s | raw: %s", e, text[:200])

        return []

    def format_for_primer(self, hypotheses: List[Dict[str, Any]]) -> str:
        if not hypotheses:
            return (
                "## Diagnostic Hypotheses\n"
                "No hypotheses generated — proceed with manual exploration.\n"
                "Start by reading the bug report carefully and running the test command."
            )

        lines = [
            "## Diagnostic Hypotheses",
            "(Verify these leads FIRST before broad exploration — "
            "each has a ready-to-run bash command)\n",
        ]
        for i, h in enumerate(hypotheses, 1):
            lines.append(f"### Lead {i}: {h.get('title', 'Unknown')}")
            lines.append(f"- **Suspect file:** `{h.get('file_lead', '?')}`")
            lines.append(f"- **First command:** `{h.get('verify_cmd', '?')}`")
            lines.append(f"- **Reasoning:** {h.get('reasoning', '')}")
            lines.append("")

        return "\n".join(lines)