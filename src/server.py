"""
Purple Agent v4.2.2 — Phase 2 server

Pipeline:
  Stage 1   — Container bootstrap + repo detection
  Stage 2   — Test command discovery (no execution — avoids 3-4 min stalls)
  Stage 1.5 — Hypothesis generation (problem_statement + hints_text + file tree)
  Stage 3   — ICL injection
  Stage 4   — 20-turn stateful bash REPL
  Stage 5   — Mechanical test gate
  Stage 6   — Targeted QA repair (up to 2 retries)

Global 260s asyncio.wait_for wraps the entire pipeline.
Best-effort patch written to /tmp/purple_patch.diff periodically for timeout recovery.
"""

import os
import json
import uuid
import logging
import asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.core.llm_client import LLMClient
from src.core.docker_bridge import DockerBridge
from src.core.agent_loop import AgentLoop
from src.tools.test_engine import TestEngine
from src.tools.hypotheses import HypothesisGenerator
from src.prompts.icl_specialist import ICLSpecialist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("purple_agent")

PORT                = int(os.getenv("PORT",          "9022"))
GLOBAL_TASK_TIMEOUT = int(os.getenv("TASK_TIMEOUT_S", "260"))

# ── Agent card ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Purple Coding Agent (Phase 2)")

AGENT_CARD = {
    "name":        "Purple Coding Agent",
    "description": (
        "SWE-bench Phase 2: Stateful Bash REPL + Docker-out-of-Docker. "
        "Hypothesis synthesis from problem_statement + hints_text. "
        "20-turn budget with mechanical test gate."
    ),
    "url":     f"http://localhost:{PORT}/",
    "version": "4.2.3",
    "capabilities": {
        "streaming":              False,
        "pushNotifications":      False,
        "stateTransitionHistory": False,
    },
    "defaultInputModes":  ["application/json"],
    "defaultOutputModes": ["application/json"],
    "skills": [{
        "id":          "swe_patch",
        "name":        "SWE Patch",
        "description": "Bash REPL with live test execution and verified diff output.",
        "tags":        ["coding", "swe-bench", "patch", "docker"],
        "examples":    [],
    }],
}


@app.get("/.well-known/agent-card.json")
async def agent_card():
    return JSONResponse(content=AGENT_CARD)


@app.get("/.well-known/agent.json")
async def agent_card_compat():
    return JSONResponse(content=AGENT_CARD)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "4.2.3"}


# ==============================================================================
# TASK EXTRACTION
# Handles all A2A envelope formats sent by the green agent.
# Extracts ALL fields from instances.jsonl, including hints_text.
# ==============================================================================

def _extract_task(body: dict) -> tuple[dict, str]:
    """
    Parse the A2A JSON-RPC envelope and return (task_data, context_id).
    task_data includes: problem_statement, repo, docker_image, base_commit,
                        hints_text, instance_id, short_id
    """
    context_id = ""

    # Flat body (problem_statement at top level)
    if "problem_statement" in body:
        return body, context_id

    try:
        params     = body.get("params", {})
        message    = params.get("message", {})
        context_id = message.get("contextId", "") or params.get("contextId", "")

        for part in message.get("parts", []):
            kind = part.get("kind") or part.get("type", "")
            text = part.get("text", "")

            # Structured data part
            if kind == "data" and "problem_statement" in part.get("data", {}):
                return part["data"], context_id

            # JSON-in-text part
            if kind == "text" and text.strip():
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict) and "problem_statement" in parsed:
                        return parsed, context_id
                except Exception:
                    pass
                return {"problem_statement": text.strip()}, context_id

    except Exception as e:
        logger.error("Extraction error: %s", e)

    return {}, context_id


# ==============================================================================
# CORE TASK RUNNER
# ==============================================================================

async def _run_task(task_data: dict, llm: LLMClient) -> str:
    """
    Full pipeline — always returns a unified diff (may be empty).
    Designed to complete in ≤ 260s.
    """
    problem_statement = task_data.get("problem_statement", "")
    hints_text        = task_data.get("hints_text", "")        # ← from instances.jsonl
    image_name        = task_data.get("docker_image", "")
    base_commit       = task_data.get("base_commit", "HEAD")
    repo              = task_data.get("repo", "")
    instance_id       = task_data.get("instance_id", "")

    if hints_text:
        logger.info("hints_text present: %d chars", len(hints_text))

    if not image_name:
        logger.error("No docker_image in task — cannot start container")
        return ""

    docker = DockerBridge(image_name=image_name, base_commit=base_commit)
    tester = TestEngine(docker)

    # ── Stage 1: Container bootstrap ─────────────────────────────────────────
    if not await asyncio.to_thread(docker.start_container):
        logger.error("Container bootstrap failed")
        return ""

    try:
        # ── Stage 2: Test command discovery (no execution) ────────────────────
        await asyncio.to_thread(tester.discover_test_command_only)
        logger.info("Test command discovered: %s", tester.test_command or "none")

        # ── Stage 1.5: File tree + hypothesis generation ──────────────────────
        icl     = ICLSpecialist()
        hyp_gen = HypothesisGenerator(llm)

        repo_root = docker.repo_dir
        logger.info("Building file tree from repo root: %s", repo_root)
        _, tree_output = await asyncio.to_thread(
            docker.execute_command,
            (
                f"find {repo_root} -type f "
                r"\( -name '*.py' -o -name '*.js' -o -name '*.go' "
                r"-o -name '*.ts' -o -name '*.rb' -o -name '*.java' "
                r"-o -name '*.rs' -o -name '*.kt' \) "
                r"| grep -v -E '(node_modules|__pycache__|vendor|dist|build|\.git)' "
                r"| head -120"
            ),
            30,
        )
        n_files = tree_output.count("\n") + (1 if tree_output.strip() else 0)
        logger.info("File tree: %d files found", n_files)

        try:
            hyps = await asyncio.wait_for(
                hyp_gen.generate_group(
                    problem_statement=problem_statement,
                    repo_skeleton=tree_output,
                    g_size=3,
                    hints_text=hints_text,
                    docker=docker,           # ← GSRM: execute verify_cmds for reward scoring
                    repo_dir=repo_root,      # ← GSRM: repo root path inside container
                ),
                timeout=90.0,               # increased: 3 verify_cmds × 20s each + LLM call
            )
            logger.info("Hypotheses generated: %d", len(hyps))
        except asyncio.TimeoutError:
            logger.warning("Hypothesis generation timed out — continuing without")
            hyps = []

        # ── Stage 3: Build context primer ────────────────────────────────────
        icl_block = icl.get_injection(problem_statement, tree_output)
        hyp_block = hyp_gen.format_for_primer(hyps)
        test_hint = (
            f"\n## Test Command\n`{tester.test_command}`\n"
            "Run this to check your fix. Use targeted test invocation when possible.\n"
            if tester.test_command else
            "\n## Test Command\n"
            "No standard test runner detected. Explore manually and use `git diff` "
            "to confirm changes are correct.\n"
        )
        # Also inject hints_text directly into primer if non-empty
        hints_primer = ""
        if hints_text and hints_text.strip():
            hints_primer = (
                f"\n## Benchmark Hints\n"
                f"{hints_text.strip()}\n"
                "(These hints are from the benchmark annotators — use them to narrow your search)\n"
            )

        context_primer = icl_block + "\n" + hyp_block + test_hint + hints_primer

        # ── Stage 4: 20-turn bash REPL ────────────────────────────────────────
        # Pass top hypothesis verify_cmd so framework auto-runs it before turn 1
        verify_cmd = hyps[0].get("verify_cmd", "") if hyps else ""
        agent = AgentLoop(llm, docker, tester)
        _success, messages = await agent.run_stage_4_bash_repl(
            problem_statement, context_primer, verify_cmd=verify_cmd
        )

        # ── Stage 5: Mechanical test gate ─────────────────────────────────────
        gate_passed, gate_msg = await asyncio.to_thread(tester.verify_patch)
        logger.info("Test gate: %s — %s", gate_passed, gate_msg[:120])

        # ── Stage 6: Targeted QA repair ───────────────────────────────────────
        if not gate_passed:
            logger.warning("Gate failed. Entering QA phase.")
            gate_passed = await agent.run_stage_6_qa_phase(messages, max_qa_retries=2)

        # ── Always extract git diff (partial credit on gate failure) ──────────
        _, patch = await asyncio.to_thread(
            docker.execute_command,
            (
                f"cd {repo_root} && "
                r"git ls-files --others --exclude-standard "
                r"| grep -v -E '(__pycache__|\.pyc$|\.egg-info/)' "
                r"| xargs -r git add -N -- 2>/dev/null || true && "
                r"git diff HEAD"
            ),
            30,
        )
        logger.info("Patch extracted: %d chars (gate_passed=%s)", len(patch), gate_passed)

        # Write snapshot for timeout recovery
        try:
            with open("/tmp/purple_patch.diff", "w") as pf:
                pf.write(patch)
        except Exception:
            pass

        return patch

    finally:
        await asyncio.to_thread(docker.stop_container)


# ==============================================================================
# HTTP HANDLER
# ==============================================================================

@app.post("/")
async def handle_task(request: Request):
    body        = await request.json()
    jsonrpc_id  = body.get("id", str(uuid.uuid4()))
    task_id     = str(uuid.uuid4())
    artifact_id = str(uuid.uuid4())

    task_data, context_id = _extract_task(body)
    if not context_id:
        context_id = str(uuid.uuid4())

    logger.info(
        "Task received | context=%s | repo=%s | instance=%s | hints=%s",
        context_id[:20],
        task_data.get("repo", "?"),
        task_data.get("instance_id", "?")[:40],
        "yes" if task_data.get("hints_text", "").strip() else "no",
    )

    llm           = LLMClient()
    patch_content = ""

    try:
        patch_content = await asyncio.wait_for(
            _run_task(task_data, llm),
            timeout=GLOBAL_TASK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(
            "Global task timeout (%ds) — reading best-effort diff from /tmp",
            GLOBAL_TASK_TIMEOUT,
        )
        try:
            with open("/tmp/purple_patch.diff") as f:
                patch_content = f.read()
            logger.info("Recovered %d-char patch from /tmp", len(patch_content))
        except Exception:
            logger.warning("No /tmp patch file — returning empty diff")
    except Exception as e:
        logger.exception("Unhandled task error: %s", e)

    logger.info("Response: patch_len=%d", len(patch_content))

    return JSONResponse(content={
        "jsonrpc": "2.0",
        "id":      jsonrpc_id,
        "result": {
            "id":        task_id,
            "contextId": context_id,
            "status":    {"state": "completed"},
            "artifacts": [{
                "artifactId": artifact_id,
                "name":       "patch",
                "parts":      [{"kind": "text", "text": patch_content}],
            }],
        },
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")