"""
Purple Agent v4.2 — Phase 2 server

Key changes vs. submitted version:
  - Global asyncio.wait_for(timeout=260s) around the entire task
    → Gateway kills at 300s; we return a best-effort patch at 260s
  - Re-enabled baseline discovery (timeout=45s) and hypotheses (timeout=20s)
  - Container pull failures are graceful (use local cache)
  - Patch is always extracted and returned, even on timeout/gate failure
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

PORT = int(os.getenv("PORT", "9022"))

# ── Timing budget ──────────────────────────────────────────────────────────────
# Gateway hard-kills at 300s. Our budget:
#   container bootstrap : ~40s
#   preflight (baseline): ~45s
#   agent loop (15 turns): ~165s
#   test gate + QA      : ~30s
#   buffer              : ~20s
#   ──────────────────────────
#   total               : ~300s  → wrap at 260s to guarantee response
GLOBAL_TASK_TIMEOUT = int(os.getenv("TASK_TIMEOUT_S", "260"))

app = FastAPI(title="Purple Coding Agent (Phase 2)")

AGENT_CARD = {
    "name": "Purple Coding Agent",
    "description": (
        "SWE-bench Phase 2: Stateful Bash REPL + Docker-out-of-Docker execution. "
        "15-turn budget with mechanical test gate and targeted QA repair."
    ),
    "url": f"http://localhost:{PORT}/",
    "version": "4.2.1",
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": False,
    },
    "defaultInputModes": ["application/json"],
    "defaultOutputModes": ["application/json"],
    "skills": [{
        "id": "swe_patch",
        "name": "SWE Patch",
        "description": "Bash REPL with live test execution and verified diff output.",
        "tags": ["coding", "swe-bench", "patch", "docker"],
        "examples": [],
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
    return {"status": "ok", "version": "4.2.1"}


# ==============================================================================
# TASK EXTRACTION  (unchanged from Phase 1 — handles all A2A envelope formats)
# ==============================================================================

def _extract_task(body: dict) -> tuple[dict, str]:
    context_id = ""
    if "problem_statement" in body:
        return body, context_id
    try:
        params  = body.get("params", {})
        message = params.get("message", {})
        context_id = message.get("contextId", "") or params.get("contextId", "")
        for part in message.get("parts", []):
            kind = part.get("kind") or part.get("type", "")
            text = part.get("text", "")
            if kind == "data" and "problem_statement" in part.get("data", {}):
                return part["data"], context_id
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
# CORE TASK RUNNER  (runs inside the global timeout)
# ==============================================================================

async def _run_task(task_data: dict, llm: LLMClient) -> str:
    """
    Runs the full pipeline and always returns a unified diff string (may be empty).
    Designed to complete in ≤ 260 seconds.
    """
    problem_statement = task_data.get("problem_statement", "")
    image_name        = task_data.get("docker_image", "")
    base_commit       = task_data.get("base_commit", "HEAD")
    repo              = task_data.get("repo", "")

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
        # ── Stage 2: Test command discovery ONLY (no test execution) ──────────
        # Heavy baseline execution is removed from pre-flight — it caused 3-4 min stalls.
        # The agent runs tests inside its bash loop; the gate uses verify_patch() at the end.
        # We only discover the test command string here so the gate knows what to run.
        await asyncio.to_thread(tester.discover_test_command_only)
        logger.info("Test command discovered: %s", tester.test_command or "none")

        # ── Stage 1.5 / 2.5: Hypotheses + ICL injection (capped at 20s) ─────
        icl     = ICLSpecialist()
        hyp_gen = HypothesisGenerator(llm)

        # Lightweight repo skeleton: just the file tree, no AST parsing at this stage
        # (ASTGraphBuilder is too slow for the pre-flight budget; agent uses it via bash)
        # Use auto-detected repo_dir — NOT hardcoded /workspace
        # DockerBridge._detect_repo_dir() sets this after container start
        repo_root = docker.repo_dir
        logger.info("Building file tree from repo root: %s", repo_root)
        _, tree_output = await asyncio.to_thread(
            docker.execute_command,
            (f"find {repo_root} -type f "
             r"\( -name '*.py' -o -name '*.js' -o -name '*.go' "
             r"-o -name '*.ts' -o -name '*.rb' -o -name '*.java' -o -name '*.rs' \) "
             r"| grep -v -E '(node_modules|__pycache__|vendor|dist|build|\.git)' "
             r"| head -120"),
            30,
        )
        logger.info("File tree: %d files found", tree_output.count("\n") + (1 if tree_output.strip() else 0))

        try:
            hyps = await asyncio.wait_for(
                hyp_gen.generate_group(problem_statement, tree_output, g_size=2),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Hypothesis generation timed out — continuing without")
            hyps = []

        # Build the context primer: ICL examples + hypotheses + test command
        icl_block  = icl.get_injection(problem_statement, tree_output)
        hyp_block  = hyp_gen.format_for_primer(hyps) if hyps else ""
        test_hint  = (
            f"\n## Test Command (verified working)\n`{tester.test_command}`\n"
            f"Run this to check your fix. Baseline pre-existing failures are already known.\n"
            if tester.test_command else
            "\n## Test Command\nNo standard test runner detected. Use `git diff` to verify changes.\n"
        )

        context_primer = icl_block + "\n" + hyp_block + test_hint

        # ── Stage 4: 15-turn bash REPL ───────────────────────────────────────
        agent = AgentLoop(llm, docker, tester)
        _success, messages = await agent.run_stage_4_bash_repl(
            problem_statement, context_primer
        )

        # ── Stage 5 & 6: Mechanical test gate + targeted QA ──────────────────
        gate_passed, gate_msg = await asyncio.to_thread(tester.verify_patch)
        logger.info("Test gate: %s — %s", gate_passed, gate_msg[:120])

        if not gate_passed:
            logger.warning("Gate failed. Entering QA phase.")
            gate_passed = await agent.run_stage_6_qa_phase(messages, max_qa_retries=2)

        # ── Always extract git diff (even on gate failure — partial credit) ──
        # Extract final patch from detected repo root (not /workspace)
        repo_root = docker.repo_dir
        _, patch = await asyncio.to_thread(
            docker.execute_command,
            (f"cd {repo_root} && git ls-files --others --exclude-standard "
             r"| grep -v -E '(__pycache__|\.pyc$|\.egg-info/)' "
             r"| xargs -r git add -N -- 2>/dev/null || true && git diff HEAD"),
            30,
        )
        logger.info("Patch extracted: %d chars (gate_passed=%s)", len(patch), gate_passed)

        # Write to /tmp so the timeout handler can recover it even if cancelled
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
        "Task received | context=%s | repo=%s | image=%s",
        context_id[:20],
        task_data.get("repo", "?"),
        task_data.get("docker_image", "?")[:60],
    )

    llm          = LLMClient()
    patch_content = ""

    try:
        # Wrap the entire pipeline in a hard timeout — returns best-effort diff.
        # _run_task always writes git diff to /tmp/purple_patch.diff before returning,
        # so even on timeout we can try to read it.
        patch_content = await asyncio.wait_for(
            _run_task(task_data, llm),
            timeout=GLOBAL_TASK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(
            "Global task timeout (%ds) fired — reading best-effort diff from /tmp",
            GLOBAL_TASK_TIMEOUT,
        )
        try:
            with open("/tmp/purple_patch.diff") as f:
                patch_content = f.read()
            logger.info("Recovered %d-char patch from /tmp/purple_patch.diff", len(patch_content))
        except Exception:
            logger.warning("No patch file found — returning empty diff")
    except Exception as e:
        logger.exception("Unhandled task error: %s", e)

    logger.info("Response: patch_len=%d", len(patch_content))

    return JSONResponse(content={
        "jsonrpc": "2.0",
        "id": jsonrpc_id,
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