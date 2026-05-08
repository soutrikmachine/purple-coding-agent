import json
import logging
import re
from typing import List, Dict, Any
from ..core.llm_client import LLMClient

logger = logging.getLogger(__name__)

class HypothesisGenerator:
    """
    Implements Group-Sampling to provide multiple competing debugging paths.
    """
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def generate_group(self, problem_statement: str, repo_skeleton: str, g_size: int = 3) -> List[Dict[str, Any]]:
        system_prompt = (
            "You are a diagnostic engine for a stateful engineering agent.\n"
            "Given a bug report and a repository skeleton, identify likely failure points.\n"
            "Focus on providing actionable commands to verify the bug natively.\n\n"
            "Return a JSON array containing EXACTLY your hypotheses.\n"
            "Format each object strictly as:\n"
            "{\n"
            "  \"title\": \"Short description\",\n"
            "  \"file_lead\": \"path/to/suspect/file.py\",\n"
            "  \"verify_cmd\": \"exact bash command to run first (e.g., pytest tests/...)\",\n"
            "  \"reasoning\": \"Why this is the likely cause\"\n"
            "}"
        )

        user_prompt = (
            f"Bug Report: {problem_statement}\n\n"
            f"Repo Skeleton:\n{repo_skeleton[:15000]}\n\n" # Truncate massive repos if needed
            f"Generate {g_size} distinct debugging hypotheses in JSON format."
        )

        raw_response = await self.llm.generate_step(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.7 # Higher temp for diverse group sampling
        )

        return self._parse_hypotheses(raw_response)

    def _parse_hypotheses(self, text: str) -> List[Dict[str, Any]]:
        try:
            text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
            match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
                    return parsed
        except Exception as e:
            logger.error(f"Failed to parse hypothesis group: {e}\nRaw Output: {text[:200]}")
        return []

    def format_for_primer(self, hypotheses: List[Dict[str, Any]]) -> str:
        if not hypotheses:
            return "## Initial Diagnostic Hypotheses\nNo hypotheses generated. Proceed with manual exploration."
            
        primer = ["## Initial Diagnostic Hypotheses", "(Try verifying these leads first before blind searching)"]
        for i, h in enumerate(hypotheses, 1):
            primer.append(f"\n### Strategy {i}: {h.get('title', 'Unknown')}")
            primer.append(f"- **Root Cause Lead:** `{h.get('file_lead', '')}`")
            primer.append(f"- **Verification Command:** `{h.get('verify_cmd', '')}`")
            primer.append(f"- **Reasoning:** {h.get('reasoning', '')}")
        
        return "\n".join(primer)