"""
Purple Agent A2A Server
=======================
FastAPI server implementing the A2A (Agent-to-Agent) protocol.
The green agent (AgentSWE/SWE-bench) sends tasks here; we return actions.

Protocol:
  GET  /.well-known/agent.json  -> Agent Card
  POST /                        -> Handle A2A task messages
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent import PurpleAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("purple_agent.server")

# ── Globals ──────────────────────────────────────────────────────────────────

_agent: PurpleAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    logger.info("🟣 Purple Agent starting up…")
    _agent = PurpleAgent(
        model_base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000"),
        model_name=os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"),
        max_turns=int(os.getenv("MAX_TURNS", "15")),
        mcts_branches=int(os.getenv("MCTS_BRANCHES", "3")),
        temperature=float(os.getenv("TEMPERATURE", "0.6")),
        use_mcts=os.getenv("USE_MCTS", "true").lower() == "true",
    )
    logger.info("🟣 Purple Agent ready")
    yield
    logger.info("🟣 Purple Agent shutting down")


app = FastAPI(title="Purple Coding Agent", lifespan=lifespan)

# ── Agent Card ────────────────────────────────────────────────────────────────

AGENT_CARD = {
    "name": "Purple Coding Agent",
    "description": (
        "An MCTS-guided software engineering agent that explores repositories "
        "and patches bugs using DeepSeek-Coder reasoning. "
        "Built for SWE-bench / AgentBeats Phase 2."
    ),
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
            "description": "Explore a codebase and produce a git patch to fix a bug.",
            "tags": ["coding", "debugging", "patch", "swe-bench"],
            "examples": [],
        }
    ],
}


@app.get("/.well-known/agent.json")
async def agent_card():
    return JSONResponse(content=AGENT_CARD)


# ── A2A Task Handler ──────────────────────────────────────────────────────────

@app.post("/")
async def handle_task(request: Request):
    """
    Main A2A endpoint. The green agent sends a JSON body following the A2A spec.
    We run our MCTS+LLM loop and return the action response.
    """
    body = await request.json()
    logger.info("📨 Received A2A message: %s", json.dumps(body)[:300])

    task_id = body.get("id", str(uuid.uuid4()))
    message = _extract_message(body)

    if message is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Could not parse task message"},
        )

    # Run agent (async-safe wrapper around sync reasoning loop)
    loop = asyncio.get_event_loop()
    response_action = await loop.run_in_executor(None, _agent.respond, message)

    # Wrap in A2A response envelope
    a2a_response = {
        "id": task_id,
        "result": {
            "status": {"state": "completed"},
            "artifacts": [
                {
                    "name": "action",
                    "parts": [{"type": "text", "text": json.dumps(response_action)}],
                }
            ],
        },
    }
    logger.info("📤 Sending action: %s", json.dumps(response_action)[:200])
    return JSONResponse(content=a2a_response)


def _extract_message(body: dict) -> dict | None:
    """
    Parse A2A envelope to get the actual task payload.
    Handles both direct message format and nested params/message format.
    """
    # Direct task data (green agent may send flat JSON)
    if "problem_statement" in body:
        return body

    # A2A spec: params -> message -> parts
    try:
        parts = body["params"]["message"]["parts"]
        for part in parts:
            if part.get("type") == "text":
                text = part["text"]
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"problem_statement": text}
    except (KeyError, TypeError):
        pass

    # Fallback: try to find any dict with problem_statement
    for key in ("message", "data", "input", "task"):
        if key in body and isinstance(body[key], dict):
            if "problem_statement" in body[key]:
                return body[key]

    return None


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "agent": "purple"}


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "9010"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
