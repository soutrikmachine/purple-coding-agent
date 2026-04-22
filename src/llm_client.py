"""
LLMClient
=========
Thin wrapper around OpenAI-compatible API (served by vLLM on Kaggle)
with a fallback to HuggingFace Transformers for offline use.

vLLM server is started separately (see scripts/start_vllm.sh).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger("purple_agent.llm")

# ── OpenAI-Compatible Client ──────────────────────────────────────────────────


class LLMClient:
    """
    Calls a vLLM-served OpenAI-compatible /v1/chat/completions endpoint.
    Falls back to direct HuggingFace inference if base_url is "local".
    """

    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api", # Safer default for standard /v1/ routing
        model: str = "Qwen/Qwen2.5-Coder-7B-Instruct", # Purged DeepSeek
        api_key: str = "", # New API Key parameter
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._local_model = None  # lazy-loaded HF model fallback

        if base_url == "local":
            logger.info("LLM mode: local HuggingFace (Unsloth)")
        else:
            logger.info("LLM mode: API at %s model=%s", base_url, model)

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

        # Securely inject the API key into the HTTP Headers
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/v1/chat/completions"

        for attempt in range(1, self.max_retries + 1):
            try:
                # Pass the headers parameter to the request
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
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
                logger.warning("LLM call attempt %d failed: %s", attempt, e)
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                else:
                    raise RuntimeError(f"LLM call failed after {self.max_retries} attempts") from e

        return ""  # unreachable

    # ── Local HuggingFace Fallback ────────────────────────────────────────────

    def _complete_local(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Direct inference using Unsloth-loaded model.
        Used inside the Kaggle notebook when vLLM isn't running.
        """
        if self._local_model is None:
            self._load_local_model()

        model, tokenizer = self._local_model

        # Apply chat template
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
        logger.info("Loading local model with Unsloth…")
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model,
                max_seq_length=8192,
                load_in_4bit=False,  # Disabled to preserve exact mathematical representations
                dtype=None,  # auto
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
