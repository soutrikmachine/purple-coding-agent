import os
import re
import logging
from typing import Dict, Optional, Tuple, List
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, api_key: Optional[str] = None):
        # OpenRouter-specific configuration
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is missing.")
        
        # Using the flash model for speed in the 50-turn loop
        self.model = "deepseek/deepseek-v4-flash"
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/soutrikmachine/purple-coding-agent", # OpenRouter ranking requirement
                "X-Title": "Purple Agent Phase 2"
            }
        )

    async def generate_step(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        """
        DeepSeek-v4-flash benefits from slightly lower temperature (0.1) 
        to ensure syntax stability in the bash commands.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
                stop=["<observation>", "</observation>"]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}")
            raise

    def parse_response(self, response_text: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Extracts thought and action blocks."""
        thought_match = re.search(r'<thought>(.*?)</thought>', response_text, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else ""

        # Flexibility for various action tag formats
        action_match = re.search(r'<action(?:\s+type="([^"]+)")?>(.*?)</action>', response_text, re.DOTALL | re.IGNORECASE)
        
        if action_match:
            action_type = action_match.group(1) or "bash"  # Default to bash if type isn't specified
            action_content = action_match.group(2).strip()
            return thought, action_type, action_content
            
        return thought, None, None

    def format_observation(self, output: str, exit_code: int) -> str:
        status = "SUCCESS" if exit_code == 0 else f"FAILED (Exit {exit_code})"
        return f"<observation status=\"{status}\">\n{output}\n</observation>"