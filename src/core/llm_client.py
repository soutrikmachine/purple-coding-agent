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

    def parse_response(self, response_text: str) -> Tuple[str, Optional[str], Optional[str]]:
        # ... (keep existing regex logic from previous step)
        pass

    def format_observation(self, output: str, exit_code: int) -> str:
        status = "SUCCESS" if exit_code == 0 else f"FAILED (Exit {exit_code})"
        return f"<observation status=\"{status}\">\n{output}\n</observation>"