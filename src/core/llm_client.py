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
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
                stop=["<observation>", "</observation>"] 
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
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
            markdown_match = re.search(r'```(bash|python|sh)\n(.*?)
                ```', raw_text, re.DOTALL | re.IGNORECASE)
            if markdown_match:
                logging.warning("Extracted action from markdown block instead of XML tag.")
                action_type = markdown_match.group(1).strip().lower()
                if action_type == 'sh': action_type = 'bash'
                action_content = markdown_match.group(2).strip()

        return thought, action_type, action_content

    def format_observation(self, output: str, exit_code: int) -> str:
        status = "SUCCESS" if exit_code == 0 else f"FAILED (Exit {exit_code})"
        return f"<observation status=\"{status}\">\n{output}\n</observation>"