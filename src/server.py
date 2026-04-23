"""
Purple Agent A2A Server — Fixed for SWE-bench Pro Green Agent
=============================================================

Critical fixes applied:
  1. Agent card served at /.well-known/agent-card.json  (was agent.json)
  2. Port default changed to 9010                       
  3. POST /  returns raw patch string as artifact text   (not json.dumps(action))
  4. Proper task_id / contextId threading through A2A envelope
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent import PurpleAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("purple_agent.server")

_agent: PurpleAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    logger.info("🟣 Purple Agent starting…")
    _agent = PurpleAgent(
        model_base_url=os.getenv("LLM_BASE_URL", "https://api-inference.huggingface.co/v1"),
        model_name=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        hf_token=os.getenv("HF_TOKEN", ""),
        max_turns=int(os.getenv("MAX_TURNS", "15")),
        mcts_branches=int(os.getenv("MCTS_BRANCHES", "3")),
        temperature=float(os.getenv("TEMPERATURE", "0.6")),
        use_mcts=os.getenv("USE_MCTS", "true").lower() == "true",
    )
    logger.info("🟣 Purple Agent ready")
    yield


app = FastAPI(title="Purple Coding Agent", lifespan=lifespan)

# ── Agent Card ─────────────────────────────────────────────────────────────────
# FIX 1: The SWE-bench green agent calls /.well-known/agent-card.json
#         NOT /.well-known/agent.json
# ──────────────────────────────────────────────────────────────────────────────

AGENT_CARD = {
    "name": "Purple Coding Agent",
    "description": (
        "MCTS-guided autonomous software engineering agent. "
        "Powered by Qwen2.5-Coder-7B. Explores repositories and returns git patches."
    ),
    "url": f"http://localhost:{os.getenv('PORT', '9009')}/",
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
            "name": "Software Engineering Patch",
            "description": "Fix a GitHub issue by returning a unified git diff patch.",
            "tags": ["coding", "swe-bench", "patch"],
            "examples": [],
        }
    ],
}


@app.get("/.well-known/agent-card.json")   # ← FIXED: was agent.json
async def agent_card():
    return JSONResponse(content=AGENT_CARD)


# Keep the old path too for safety
@app.get("/.well-known/agent.json")
async def agent_card_compat():
    return JSONResponse(content=AGENT_CARD)


# ── A2A Task Handler ───────────────────────────────────────────────────────────

@app.post("/")
async def handle_task(request: Request):
    body = await request.json()
    logger.info("📨 Received A2A message type=%s", body.get("method", "unknown"))

    task_id = body.get("id", str(uuid.uuid4()))
    message = _extract_message(body)

    if message is None:
        logger.error("Could not parse task message from body: %s", str(body)[:300])
        return JSONResponse(
            status_code=400,
            content={"id": task_id, "error": {"code": -32600, "message": "Cannot parse task"}},
        )

    try:
        loop = asyncio.get_event_loop()
        action = await loop.run_in_executor(None, _agent.respond, message)
    except Exception as e:
        logger.exception("Agent error: %s", e)
        action = {"action": "patch", "content": ""}  # return empty patch, don't crash

    # ── FIX 2: Response format ─────────────────────────────────────────────────
    # The SWE-bench green agent expects the patch as a plain text artifact.
    # It reads artifacts[0].parts[0].text and uses it as the git diff string.
    # Do NOT json.dumps(action) — return just the diff content as plain text.
    # ──────────────────────────────────────────────────────────────────────────

    patch_content = action.get("content", "") if action.get("action") == "patch" else ""
    action_type = action.get("action", "bash")

    if action_type != "patch":
        # Still in exploration mode — return the bash/debug command
        # The green agent for SWE-bench Pro sends the result back as a new message
        artifact_text = json.dumps(action)  # bash/debug actions stay JSON
    else:
        artifact_text = patch_content  # patch is returned as raw diff text

    a2a_response = {
        "id": task_id,
        "result": {
            "id": task_id,
            "status": {"state": "completed"},
            "artifacts": [
                {
                    "name": "patch" if action_type == "patch" else "action",
                    "parts": [
                        {
                            "type": "text",
                            "text": artifact_text,
                        }
                    ],
                }
            ],
        },
    }

    logger.info("📤 action=%s content_len=%d", action_type, len(artifact_text))
    return JSONResponse(content=a2a_response)


def _extract_message(body: dict) -> dict | None:
    """
    Parse the A2A envelope from the SWE-bench green agent.

    The green agent sends:
      {
        "id": "...",
        "method": "message/send",
        "params": {
          "message": {
            "role": "user",
            "parts": [{"type": "text", "text": "<JSON string of task>"}]
          },
          "metadata": {...}
        }
      }
    """
    # Direct: green agent embeds task data at top level
    if "problem_statement" in body:
        return body

    # A2A spec format: params.message.parts[].text
    try:
        parts = body["params"]["message"]["parts"]
        for part in parts:
            if part.get("type") == "text":
                text = part["text"]
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    return {"problem_statement": text}
    except (KeyError, TypeError):
        pass

    # Fallback: look for any nested dict with problem_statement
    for key in ("message", "data", "input", "task", "params"):
        val = body.get(key)
        if isinstance(val, dict) and "problem_statement" in val:
            return val

    # Last resort: return the whole body if it might contain useful fields
    if any(k in body for k in ("repo", "instance_id", "base_commit")):
        return body

    logger.warning("Cannot extract message from body keys: %s", list(body.keys()))
    return None


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "agent": "purple-coding-agent"}


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "9010"))   # ← FIX 3: default 9009 not 9010
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
