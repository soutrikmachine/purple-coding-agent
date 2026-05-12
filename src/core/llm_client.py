"""
LLMClient — Phase 2 v4.2.3

Supports three reasoning model families via OpenRouter:
  - MiniMax  (minimax/minimax-m2.7)  : reasoning={"enabled": True}
  - Gemini   (google/gemini-3-*)     : reasoning={"effort": "medium"}
  - Claude   (anthropic/claude-*)    : reasoning={"effort": "medium"}

All three return thinking in response.choices[0].message.reasoning_details
(a list of dicts with "text" or "thinking" keys).

MiniMax-specific: reasoning_details MUST be preserved and passed back in
subsequent assistant messages for reasoning continuity across turns.
This is handled by storing _last_reasoning_details after every call so
agent_loop.py can include it in the messages list.

REPL parser handles:
  - Gemini's ```xml fence wrapping
  - Typeless <action> tags (treats as bash)
  - Inner ```bash code blocks (fallback)
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
        self.model_name = os.getenv("MODEL_NAME", "minimax/minimax-m2.7")
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

        # Detect model family
        name = self.model_name.lower()
        self.is_minimax = "minimax" in name
        self.is_gemini  = "gemini"  in name
        self.is_claude  = "claude"  in name or "anthropic" in name
        self.is_thinking_model = self.is_minimax or self.is_gemini or self.is_claude

        # Side-channel attributes populated after each call
        self._last_reasoning         = ""   # text summary → written to NOTES.txt
        self._last_reasoning_details = None # raw list → passed back for MiniMax continuity

        logger.info(
            "LLMClient init: model=%s thinking=%s",
            self.model_name, self.is_thinking_model
        )

    async def generate_step(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.15,
        use_thinking: bool = True,
    ) -> str:
        """
        Single LLM call with three protection layers:
          1. Reasoning config injected per model family
          2. HTML/Cloudflare firewall leak detection
          3. Empty response retry with temperature nudge
        """
        payload: Dict = {
            "model":       self.model_name,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  4096,   # must exceed thinking budget
        }

        # Inject reasoning — format differs by model family
        if self.is_thinking_model and use_thinking:
            if self.is_minimax:
                # MiniMax uses a simple boolean enable
                payload["extra_body"] = {"reasoning": {"enabled": True}}
            else:
                # Gemini 3 and Claude use effort levels (low/medium/high)
                # "medium" gives meaningful thinking without excessive latency
                payload["extra_body"] = {"reasoning": {"effort": "medium"}}

        # Reset side-channel before call
        self._last_reasoning         = ""
        self._last_reasoning_details = None

        for attempt in range(1, 3):
            try:
                response = await self.client.chat.completions.create(**payload)
                msg      = response.choices[0].message

                # ── DIAGNOSTIC LOG — remove after one run ──────────────────────
                logger.info(
                "RAW MSG: content=%d chars | reasoning_details=%s | model_extra_keys=%s",
                len(msg.content or ""),
                str(getattr(msg, "reasoning_details", "ATTR_MISSING"))[:120],
                list((getattr(msg, "model_extra", {}) or {}).keys()),
                )

                # ── Extract content ───────────────────────────────────────────
                content = msg.content or ""

                # ── Extract reasoning_details ─────────────────────────────────
                # OpenRouter normalises all thinking models to reasoning_details.
                # MiniMax REQUIRES this to be preserved and passed back in the
                # next turn's messages list — without it reasoning degrades.
                reasoning_text    = ""
                reasoning_details = None
                if self.is_thinking_model:
                    try:
                        rd = getattr(msg, "reasoning_details", None)
                        if rd and isinstance(rd, list):
                            reasoning_details = rd
                            reasoning_text = " ".join(
                                item.get("text", "") or item.get("thinking", "")
                                for item in rd
                                if isinstance(item, dict)
                            ).strip()
                        # Fallbacks for providers that surface it differently
                        if not reasoning_text:
                            reasoning_text = getattr(msg, "reasoning", None) or ""
                        if not reasoning_text:
                            extra = getattr(msg, "model_extra", {}) or {}
                            rd2 = extra.get("reasoning_details") or extra.get("reasoning") or ""
                            if isinstance(rd2, list):
                                reasoning_details = rd2
                                reasoning_text = " ".join(
                                    str(r.get("text", "") or r.get("thinking", ""))
                                    for r in rd2
                                )
                            else:
                                reasoning_text = str(rd2) if rd2 else ""
                    except Exception as e:
                        logger.debug("reasoning_details extraction error: %s", e)

                self._last_reasoning         = reasoning_text
                self._last_reasoning_details = reasoning_details

                logger.debug(
                    "LLM: %d chars content, %d chars reasoning",
                    len(content), len(reasoning_text)
                )

                # ── Firewall leak detection ────────────────────────────────────
                if content and (
                    "<html" in content.lower() or "cloudflare" in content.lower()
                ):
                    logger.error("Provider returned HTML/firewall page — retrying")
                    raise ValueError("HTML firewall leak")

                # ── Empty response retry ───────────────────────────────────────
                if not content.strip():
                    if attempt == 1:
                        logger.warning("Empty response — retrying with higher temperature")
                        payload["temperature"] = min(temperature + 0.25, 0.6)
                        continue
                    logger.error("Empty response after retry — giving up")
                    return ""

                return content

            except Exception as e:
                err = str(e)
                if len(err) > 300 or "<html" in err.lower():
                    err = "Provider error (HTML/WAF). Try a shorter command."
                logger.error("LLM attempt %d/2 failed: %s", attempt, err[:150])
                if attempt == 1:
                    continue
                raise

        return ""

    def parse_response(self, raw_text: str) -> Tuple[str, str, str]:
        """
        Extracts (thought, action_type, action_content) from LLM output.

        Parsing order:
          1. Strip outer ```xml fence (Gemini pattern)
          2. Match typed  <action type="bash">...</action>
          3. Match typeless <action>...</action> → treated as bash
          4. Match inner ```bash block (DeepSeek fallback)
          5. Return safe error echo

        Always returns a valid triple — never raises.
        """
        if not raw_text or not raw_text.strip():
            return (
                "Empty response.",
                "bash",
                "echo 'EMPTY RESPONSE. Use <action type=\"bash\">cmd</action>.'",
            )

        import re as _re

        # Strip outer markdown fence (Gemini wraps entire XML in ```xml ... ```)
        text = raw_text.strip()
        fence = _re.match(
            r"^```(?:xml|json|markdown)?\s*\n(.*?)\n?```\s*$", text, _re.DOTALL
        )
        if fence:
            logger.debug("Stripped outer markdown fence")
            text = fence.group(1).strip()

        thought = "No thought provided."
        action_type = ""
        action_content = ""

        # Extract <thought>
        m = _re.search(r"<thought>(.*?)</thought>", text, _re.DOTALL | _re.IGNORECASE)
        if m:
            thought = m.group(1).strip()

        # Extract typed <action type="...">
        m = _re.search(
            r"<action\s+type=['\"]?(\w+)['\"]?>(.*?)</action>",
            text,
            _re.DOTALL | _re.IGNORECASE,
        )
        if m:
            action_type    = m.group(1).strip().lower()
            action_content = m.group(2).strip()
        else:
            # Typeless <action> → bash
            m = _re.search(r"<action>(.*?)</action>", text, _re.DOTALL | _re.IGNORECASE)
            if m:
                logger.warning("Typeless <action> — treating as bash")
                action_type    = "bash"
                action_content = m.group(1).strip()
            else:
                # Inner ```bash block (DeepSeek / non-compliant)
                m = _re.search(r"```(?:bash|sh|python)\n(.*?)```", text, _re.DOTALL)
                if m:
                    logger.warning("Extracted action from inner markdown block")
                    action_type    = "bash"
                    action_content = m.group(1).strip()
                else:
                    action_type    = "bash"
                    action_content = (
                        "echo 'PARSE ERROR: No action tag. "
                        "Use <action type=\"bash\">cmd</action>. NO markdown fences.'"
                    )

        if action_type == "sh":
            action_type = "bash"

        return thought, action_type, action_content

    @staticmethod
    def format_observation(output: str, exit_code: int) -> str:
        status = "SUCCESS" if exit_code == 0 else f"FAILED (exit {exit_code})"
        return f'<observation status="{status}">\n{output}\n</observation>'