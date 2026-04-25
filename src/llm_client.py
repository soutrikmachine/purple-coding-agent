"""
LLMClient
=========
Thin wrapper around OpenAI-compatible API.
Supports HuggingFace Router, OpenRouter, or any vLLM-served endpoint.

Fixes applied:
  1. No double /v1 in URL — base_url already contains /v1
  2. Token read from env vars as fallback (HF_TOKEN or LLM_API_KEY)
  3. Returns empty string on failure instead of crashing
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger("purple_agent.llm")


class LLMClient:
    """
    Calls an OpenAI-compatible /chat/completions endpoint.
    Works with HuggingFace Router, OpenRouter, or vLLM.
    """

    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "deepseek/deepseek-v3.2",
        api_key: str = "",
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._local_model = None

        # FIX 2: read token from argument first, then fall back to env vars
        self.api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY", "")
            or os.getenv("LLM_API_KEY", "")
            or os.getenv("HF_TOKEN", "")
        )

        if self.api_key:
            logger.info("LLM auth token set ✓")
        else:
            logger.warning("No API token found — LLM calls will fail with 401")

        # FIX 1: build the correct chat completions URL once
        # base_url may or may not end in /v1 — handle both cases
        if self.base_url.endswith("/v1"):
            self._chat_url = f"{self.base_url}/chat/completions"
        else:
            self._chat_url = f"{self.base_url}/v1/chat/completions"

        logger.info("LLM endpoint: %s  model: %s", self._chat_url, self.model)

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.6,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> str:
        if self.base_url == "local":
            return self._complete_local(messages, temperature, max_tokens)
        return self._complete_api(messages, temperature, max_tokens, stop)

    # ── API Path ──────────────────────────────────────────────────────────────

    def _complete_api(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self._chat_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )

                if resp.status_code == 401:
                    logger.error(
                        "LLM returned 401 Unauthorized. "
                        "Check that OPENROUTER_API_KEY is set correctly in amber-manifest.json5 secrets."
                    )
                    return ""

                if resp.status_code == 429:
                    logger.warning("LLM rate limited (429). Waiting before retry...")
                    time.sleep(5 * attempt)
                    continue

                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                logger.debug(
                    "LLM: %d prompt + %d completion tokens",
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                )
                return content

            except requests.RequestException as e:
                logger.warning("LLM attempt %d/%d failed: %s", attempt, self.max_retries, e)
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                else:
                    # FIX 3: return empty string instead of raising — agent will
                    # return empty patch, task scores 0 but container doesn't crash
                    logger.error("All LLM retries exhausted, returning empty string")
                    return ""

        return ""

    # ── Local HuggingFace Fallback ────────────────────────────────────────────

    def _complete_local(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Direct inference using locally loaded model.
        Only used when LLM_BASE_URL=local.
        """
        if self._local_model is None:
            self._load_local_model()

        model, tokenizer = self._local_model

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _load_local_model(self):
        logger.info("Loading local model...")
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model,
                max_seq_length=8192,
                load_in_4bit=True,
                dtype=None,
            )
            FastLanguageModel.for_inference(model)
        except ImportError:
            logger.warning("Unsloth not available, falling back to transformers")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        self._local_model = (model, tokenizer)
        logger.info("Local model loaded ✓")
