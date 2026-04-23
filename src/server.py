"""
Purple Agent — Simplified Single-Shot Server
=============================================
Stripped down to the bare minimum for reliability:
  - One LLM call per task
  - No MCTS, no sessions, no multi-turn complexity
  - Returns a patch directly
  - Heavy logging so GitHub Actions output shows what's happening

If this gets 0 passes, the problem is definitely the LLM API, not the agent logic.
"""

import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("purple_agent")

# ── Config from env ───────────────────────────────────────────────────────────

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")
API_KEY      = (
    os.getenv("HF_TOKEN", "")
    or os.getenv("LLM_API_KEY", "")
    or os.getenv("OPENROUTER_API_KEY", "")
)
PORT         = int(os.getenv("PORT", "9010"))

# Build correct URL — avoid double /v1
if "/v1" in LLM_BASE_URL:
    CHAT_URL = f"{LLM_BASE_URL}/chat/completions"
else:
    CHAT_URL = f"{LLM_BASE_URL}/v1/chat/completions"

logger.info("=" * 60)
logger.info("Purple Agent starting")
logger.info("LLM URL   : %s", CHAT_URL)
logger.info("Model     : %s", MODEL_NAME)
logger.info("API Key   : %s", "SET ✓" if API_KEY else "MISSING ✗")
logger.info("Port      : %d", PORT)
logger.info("=" * 60)

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Purple Coding Agent")

AGENT_CARD = {
    "name": "Purple Coding Agent",
    "description": "Autonomous software engineering agent for SWE-bench Pro.",
    "url": f"http://localhost:{PORT}/",
    "version": "1.0.0",
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": False,
    },
    "defaultInputModes": ["application/json"],
    "defaultOutputModes": ["application/json"],
    "skills": [
        {
            "id": "swe_patch",
            "name": "SWE Patch",
            "description": "Fix a GitHub issue and return a unified git diff patch.",
            "tags": ["coding", "swe-bench", "patch"],
            "examples": [],
        }
    ],
}


@app.get("/.well-known/agent-card.json")
async def agent_card():
    return JSONResponse(content=AGENT_CARD)


@app.get("/.well-known/agent.json")
async def agent_card_compat():
    return JSONResponse(content=AGENT_CARD)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/")
async def handle_task(request: Request):
    body = await request.json()
    task_id = body.get("id", str(uuid.uuid4()))

    logger.info("─" * 50)
    logger.info("Received task id=%s", task_id)

    # Extract problem statement from A2A envelope
    problem_statement = _extract_problem(body)
    logger.info("Problem statement (first 200 chars): %s", problem_statement[:200])

    if not problem_statement:
        logger.error("Could not extract problem statement from body")
        return _make_response(task_id, "")

    # Call LLM
    patch = _call_llm(problem_statement)
    logger.info("Patch returned (first 300 chars): %s", patch[:300] if patch else "(empty)")

    return _make_response(task_id, patch)


# ── LLM Call ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert software engineer. You will be given a GitHub issue description.
Your job is to produce a git unified diff patch that fixes the issue.

Rules:
- Output ONLY the patch in unified diff format (starting with diff --git)
- Do not include any explanation or markdown
- The patch must be valid and directly applicable with `git apply`
- Make minimal, targeted changes
"""

def _call_llm(problem_statement: str) -> str:
    if not API_KEY:
        logger.error("No API key — skipping LLM call")
        return ""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Fix this issue:\n\n{problem_statement}"},
    ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    logger.info("Calling LLM at %s ...", CHAT_URL)

    for attempt in range(1, 4):
        try:
            resp = requests.post(CHAT_URL, json=payload, headers=headers, timeout=120)
            logger.info("LLM HTTP status: %d", resp.status_code)

            if resp.status_code == 401:
                logger.error("401 Unauthorized — API key is wrong or missing")
                return ""

            if resp.status_code == 429:
                logger.warning("429 Rate limited — waiting %ds", 5 * attempt)
                import time; import time as t; t.sleep(5 * attempt)
                continue

            if resp.status_code != 200:
                logger.error("LLM error %d: %s", resp.status_code, resp.text[:300])
                return ""

            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            logger.info("LLM raw response length: %d chars", len(raw))

            # Extract patch from response
            patch = _extract_patch(raw)
            logger.info("Extracted patch length: %d chars", len(patch))
            return patch

        except Exception as e:
            logger.warning("LLM attempt %d failed: %s", attempt, e)
            import time; time.sleep(2 ** attempt)

    logger.error("All LLM attempts failed")
    return ""


def _extract_patch(raw: str) -> str:
    """Extract unified diff from LLM response."""
    raw = raw.strip()

    # Strip markdown code fences if present
    if "```" in raw:
        # Try to extract content between fences
        match = re.search(r"```(?:diff|patch)?\n(.*?)```", raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()

    # Check if it looks like a patch
    if raw.startswith("diff --git") or raw.startswith("--- "):
        return raw

    # Try to find a diff block anywhere in the response
    diff_match = re.search(r"(diff --git.*)", raw, re.DOTALL)
    if diff_match:
        return diff_match.group(1).strip()

    logger.warning("LLM response does not contain a recognizable diff")
    logger.info("Raw LLM output: %s", raw[:500])
    return raw  # return as-is and let the green agent handle it


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_problem(body: dict) -> str:
    """Extract problem statement from A2A message envelope."""

    # Direct field
    if "problem_statement" in body:
        return body["problem_statement"]

    # A2A spec: params.message.parts[].text
    try:
        parts = body["params"]["message"]["parts"]
        for part in parts:
            if part.get("type") == "text":
                text = part["text"]
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed.get("problem_statement", text)
                except (json.JSONDecodeError, ValueError):
                    return text
    except (KeyError, TypeError):
        pass

    # Fallback: search nested dicts
    for key in ("message", "data", "input", "task", "params"):
        val = body.get(key)
        if isinstance(val, dict):
            if "problem_statement" in val:
                return val["problem_statement"]

    logger.warning("Could not find problem_statement in keys: %s", list(body.keys()))
    return ""


def _make_response(task_id: str, patch: str) -> JSONResponse:
    return JSONResponse(content={
        "id": task_id,
        "result": {
            "id": task_id,
            "status": {"state": "completed"},
            "artifacts": [
                {
                    "name": "patch",
                    "parts": [{"type": "text", "text": patch}],
                }
            ],
        },
    })


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, log_level="info")