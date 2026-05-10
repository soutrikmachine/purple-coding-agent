"""
LLMClient — Phase 2 v4.2.1

Key fixes vs. submitted version:
  - reasoning_content fallback: DeepSeek-v4-flash returns null content when
    it enters reasoning mode — we now check reasoning_content as fallback
  - Context pruning removed from here (AgentLoop._prune_context owns it)
  - Provider routing: Parasail/NovitaAI for DeepSeek; auto-route for Gemini
  - max_tokens reduced: 2048 (was 4096 — 4096 caused some providers to reject)
  - stop tokens removed (caused premature truncation on multi-line diffs)
  - Empty response retry: temperature nudge 0.15 → 0.4 (not 0.7 — too noisy)
"""

import os
import re
import logging
from typing import Dict, List, Tuple

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.base_url   = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        # OpenRouter slug for Gemini 3 Flash Preview.
        # Verify at: https://openrouter.ai/models — search "gemini"
        # Common slugs: google/gemini-2.5-flash-preview or google/gemini-3-flash-preview
        self.model_name = os.getenv("MODEL_NAME", "google/gemini-2.5-flash-preview")
        self.api_key    = (
            os.getenv("OPENROUTER_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("HF_TOKEN")
        )

        if not self.api_key:
            raise ValueError("No API key found. Set OPENROUTER_API_KEY.")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/soutrikmachine/purple-coding-agent",
                "X-Title": "Purple Agent Phase 2",
            },
        )
        self._last_reasoning = ""   # populated after each Gemini call
        logger.info(
            "LLMClient init: model=%s base_url=%s", self.model_name, self.base_url
        )

    async def generate_step(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.15,
    ) -> str:
        """
        Single LLM call with three layers of protection:
        1. reasoning_content fallback  (DeepSeek null-content fix)
        2. HTML/firewall detection     (Cloudflare leak guard)
        3. Empty response retry        (nudge temperature, one retry)
        """
        is_gemini = "gemini" in self.model_name.lower()

        payload = {
            "model":       self.model_name,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  2048,
            # NO stop tokens — they truncate multi-line patches mid-way
            # Provider routing is model-family specific:
            # - DeepSeek → Parasail/NovitaAI (dedicated, fastest path)
            # - Gemini   → let OpenRouter auto-route to Google AI Platform
            #              (Parasail/NovitaAI don't host Gemini; forcing them
            #               adds 2–5s wasted fallback latency per turn)
            "extra_body": {
                "provider": (
                    {
                        "order":           ["Parasail", "NovitaAI"],
                        "allow_fallbacks": True,
                        "ignore":          ["AtlasCloud"],
                    }
                    if not is_gemini else
                    {
                        # No "order" — direct route to Google AI Platform
                        "ignore": ["AtlasCloud"],
                    }
                ),
                # NOTE: Do NOT pass thinking/budget_tokens for Gemini.
                # Gemini 2.5/3 Flash thinks automatically — no explicit enable needed.
                # {"thinking": ...} is the Anthropic Claude format; Gemini via
                # OpenRouter ignores or rejects it. reasoning_content is populated
                # automatically and captured via the side-channel in _last_reasoning.
            },
        }

        for attempt in range(1, 3):
            try:
                response = await self.client.chat.completions.create(**payload)
                msg = response.choices[0].message

                # Layer 1: Extract content + reasoning (Gemini-safe)
                content = msg.content

                # For Gemini: reasoning_content holds chain-of-thought.
                # We do NOT use it as a content replacement (it has no XML tags).
                # Instead, return it as a side-channel so agent_loop can write
                # it to NOTES.txt for long-term reasoning memory.
                # For DeepSeek: DO NOT fall back to reasoning_content — it breaks REPL.
                reasoning = ""
                if is_gemini:
                    reasoning = getattr(msg, "reasoning_content", None) or ""

                # Layer 2: HTML/firewall leak detection
                if content and ("<html" in content.lower() or "cloudflare" in content.lower()):
                    logger.error("Provider returned HTML page instead of LLM text")
                    raise ValueError("Provider HTML leak — retrying")

                # Layer 3: empty response retry
                if not content or not content.strip():
                    if attempt == 1:
                        logger.warning("Empty response — retrying with higher temperature")
                        payload["temperature"] = min(temperature + 0.25, 0.6)
                        continue
                    else:
                        logger.error("Empty response after retry — giving up")
                        return ""

                logger.debug(
                    "LLM response: %d chars (reasoning: %d chars)",
                    len(content), len(reasoning)
                )
                # Store reasoning in a side-channel attribute the caller can read
                # without it polluting the message history.
                self._last_reasoning = reasoning
                return content

            except Exception as e:
                err = str(e)
                # Sanitize huge HTML errors from being injected into context
                if len(err) > 300 or "<html" in err.lower():
                    err = "Provider error (HTML/WAF). Try a shorter command."
                logger.error("LLM attempt %d/2 failed: %s", attempt, err[:150])
                if attempt == 1:
                    continue
                raise

        return ""

    def parse_response(self, raw_text: str) -> tuple:
        """
        Extracts (thought, action_type, action_content) from LLM output.
        Handles Gemini's outer ```xml fences and DeepSeek's inner ```bash blocks.
        Always returns a valid triple - never raises.
        """
        if not raw_text or not raw_text.strip():
            return (
                "Empty response.",
                "bash",
                "echo 'EMPTY RESPONSE. Use <action type=\"bash\">cmd</action>.'",
            )

        import re as _re

        # Gemini wraps its entire XML in ```xml ... ``` fences - strip them first
        text = raw_text.strip()
        fence = _re.match(
            r"^```(?:xml|json|markdown)?\s*\n(.*?)\n?```\s*$",
            text,
            _re.DOTALL,
        )
        if fence:
            logger.debug("Stripped outer markdown fence from Gemini response")
            text = fence.group(1).strip()

        thought = "No thought provided."
        action_type = ""
        action_content = ""

        # Extract <thought>
        m = _re.search(r"<thought>(.*?)</thought>", text, _re.DOTALL | _re.IGNORECASE)
        if m:
            thought = m.group(1).strip()

        # Extract <action type="...">
        m = _re.search(
            r"<action\s+type=['\"]?(\w+)['\"]?>(.*?)</action>",
            text,
            _re.DOTALL | _re.IGNORECASE,
        )
        if m:
            action_type    = m.group(1).strip().lower()
            action_content = m.group(2).strip()
        else:
            # Fallback 1: typeless <action> tag (model forgot type= attribute)
            m = _re.search(r"<action>(.*?)</action>", text, _re.DOTALL | _re.IGNORECASE)
            if m:
                logger.warning("Typeless <action> tag — treating as bash")
                action_type    = "bash"
                action_content = m.group(1).strip()
            else:
                # Fallback 2: inner ```bash block (DeepSeek / non-compliant)
                m = _re.search(r"```(?:bash|sh|python)\n(.*?)```", text, _re.DOTALL)
                if m:
                    logger.warning("Extracted action from inner markdown block (not XML)")
                    action_type    = "bash"
                    action_content = m.group(1).strip()
                else:
                    action_type    = "bash"
                    action_content = (
                        "echo 'PARSE ERROR: No action tag found. "
                        "Respond with <thought>...</thought><action type=\"bash\">cmd</action>. "
                        "NO markdown fences.'"
                    )


        if action_type == "sh":
            action_type = "bash"

        return thought, action_type, action_content
    @staticmethod
    def format_observation(output: str, exit_code: int) -> str:
        status = "SUCCESS" if exit_code == 0 else f"FAILED (exit {exit_code})"
        return f'<observation status="{status}">\n{output}\n</observation>'