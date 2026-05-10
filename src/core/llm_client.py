import os
import re
import logging
from typing import Dict, Optional, Tuple, List
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        # Restore Phase 1 flexibility for URLs and Models
        self.base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.model_name = os.getenv("MODEL_NAME", "deepseek/deepseek-v4-flash")
        
        # Priority-ordered API Key extraction
        self.api_key = (
            os.getenv("OPENROUTER_API_KEY") or 
            os.getenv("LLM_API_KEY") or 
            os.getenv("HF_TOKEN")
        )
        
        if not self.api_key:
            raise ValueError("No LLM API Key found in environment variables.")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/soutrikmachine/purple-coding-agent",
                "X-Title": "Purple Agent Phase 2"
            }
        )

    async def generate_step(self, messages: List[Dict[str, str]], temperature: float = 0.15) -> str:
        """
        Generates a single turn with multi-layer protection:
        1. Context Pruning (Stops Token Explosion)
        2. Provider Routing (Blocks AtlasCloud)
        3. HTML Error Detection (Paranoid Shield)
        4. Empty Response Recovery
        """
        
        # --- LAYER 1: THE CONTEXT SQUEEZER ---
        # If the history is too long, we keep the System prompt (0), 
        # the Problem Statement (1), and only the most recent 12 turns.
        if len(messages) > 12:
            logger.warning(f"Context saturated ({len(messages)} turns). Pruning to save tokens and IQ.")
            system_msg = messages[0]
            task_msg = messages[1]
            # Keeping the last 10 messages (5 turns of thought/action/observation)
            recent_context = messages[-10:]
            messages = [system_msg, task_msg] + recent_context

        try:
            # --- LAYER 2: THE PROVIDER SHIELD ---
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
                stop=["<observation>", "</observation>"],
                extra_body={
                    "provider": {
                        "order": ["DeepInfra","Parasail", "NovitaAI"],
                        "allow_fallbacks": True,
                        "ignore": ["AtlasCloud"]  # Banned for aggressive WAF
                    }
                }
            )
            
            content = response.choices[0].message.content

            # --- LAYER 3: THE PARANOID SHIELD ---
            # Catch cases where a provider returns a 200 OK but the body is Cloudflare HTML
            if content and ("<html" in content.lower() or "cloudflare" in content.lower()):
                logger.error("Detected HTML/Cloudflare leak in successful API response.")
                raise ValueError("API provider returned HTML error page instead of LLM text.")

            # --- LAYER 4: EMPTY RESPONSE RECOVERY ---
            # If the model 'chokes' and returns nothing, try one high-temp 'nudge'
            if not content or not content.strip():
                logger.warning("Received empty response. Retrying with higher temperature nudge...")
                retry_response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7, # Increased entropy to force an output
                    max_tokens=4096,
                    extra_body={"provider": {"ignore": ["AtlasCloud"]}}
                )
                content = retry_response.choices[0].message.content

            return content or ""

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            # Raising lets the sanitized exception handler in agent_loop.py take over
            raise

    def parse_response(self, raw_text: str) -> tuple[str, str, str]:
        """
        Forgiving parser that extracts <thought> and <action> tags.
        Guarantees a return of (thought, action_type, action_content).
        """
        if not raw_text:
             return (
                 "Empty response received.", 
                 "bash", 
                 "echo 'System Error: Received empty response. Please provide a valid <thought> and <action>.'"
             )

        # Default fallbacks
        thought = "No thought provided."
        action_type = "bash"
        action_content = "echo 'System Error: No valid <action> tag found. You MUST format your response with <action type=\"...\">...</action>. Try again.'"

        # Extract thought (optional, but good for logging)
        # re.DOTALL ensures it matches across newlines
        thought_match = re.search(r'<thought>(.*?)</thought>', raw_text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Greedy extraction for action type and content
        # Handles optional quotes: <action type="bash"> or <action type=python>
        action_match = re.search(r'<action\s+type=[\'"]?(.*?)[\'"]?>(.*?)</action>', raw_text, re.DOTALL | re.IGNORECASE)

        if action_match:
            action_type = action_match.group(1).strip().lower()
            action_content = action_match.group(2).strip()
        else:
            # Fallback heuristic: If the LLM just dumped markdown code blocks instead of XML
            markdown_match = re.search(r'```(bash|python|sh)\n(.*?)```', raw_text, re.DOTALL | re.IGNORECASE)
            if markdown_match:
                logging.warning("Extracted action from markdown block instead of XML tag.")
                action_type = markdown_match.group(1).strip().lower()
                if action_type == 'sh': action_type = 'bash'
                action_content = markdown_match.group(2).strip()

        return thought, action_type, action_content

    def format_observation(self, output: str, exit_code: int) -> str:
        status = "SUCCESS" if exit_code == 0 else f"FAILED (Exit {exit_code})"
        return f"<observation status=\"{status}\">\n{output}\n</observation>"