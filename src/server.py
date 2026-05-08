import os
import uuid
import json
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .core.llm_client import LLMClient
from .core.docker_bridge import DockerBridge
from .core.agent_loop import AgentLoop
from .tools.ast_graph import ASTGraphBuilder
from .tools.hypotheses import HypothesisGenerator
from .tools.test_engine import TestEngine
from .prompts.icl_specialist import ICLSpecialist

# ── Logging & Config (Restored from Phase 1) ──────────────────────────[cite: 2]
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("purple_agent")

PORT = int(os.getenv("PORT", "9010"))
# Note: MCTS_BRANCHES is now handled via Group-Sampling in hypotheses.py

logger.info("=" * 60)
logger.info("Purple Agent Phase 2  model=%s", os.getenv("MODEL_NAME", "deepseek/deepseek-v4-flash"))
logger.info("A2A Port              %d", PORT)
logger.info("Docker Socket Bind    ENABLED")
logger.info("=" * 60)

app = FastAPI(title="Purple Agent Phase 2")

# ==============================================================================
# A2A HANDSHAKE (Refined for Phase 2) ──────────────────────────────[cite: 2]
# ==============================================================================

def _extract_task_and_context(body: dict) -> tuple[dict, str]:
    """Parse A2A JSON-RPC envelope used by the Green Agent[cite: 2]."""
    context_id = ""
    # Direct pass-through if not wrapped in JSON-RPC[cite: 2]
    if "problem_statement" in body:
        return body, context_id

    try:
        params = body.get("params", {})
        message = params.get("message", {})
        context_id = message.get("contextId", "") or params.get("contextId", "")
        parts = message.get("parts", [])

        for part in parts:
            kind = part.get("kind") or part.get("type", "")
            text = part.get("text", "")

            # Turn 1: Problem statement delivered as data[cite: 2]
            if kind == "data":
                data = part.get("data", {})
                if isinstance(data, dict) and "problem_statement" in data:
                    return data, context_id
            
            # Turn 2+: Observations delivered as text[cite: 2]
            if kind == "text" and text.strip():
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict) and "problem_statement" in parsed:
                        return parsed, context_id
                except:
                    pass
                return {"problem_statement": text.strip()}, context_id
    except Exception as e:
        logger.error(f"Extraction error: {e}")
    return {}, context_id

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

@app.post("/")
async def handle_task(request: Request):
    body = await request.json()
    jsonrpc_id = body.get("id", str(uuid.uuid4()))
    
    # Extract problem_statement and repo metadata[cite: 2]
    task_data, context_id = _extract_task_and_context(body)
    if not context_id:
        context_id = str(uuid.uuid4())
        
    problem_statement = task_data.get("problem_statement", "")
    repo = task_data.get("repo", "unknown/repo")
    
    # In Phase 2, we use the instance image provided or a default[cite: 2]
    image_name = task_data.get("container_image", "swe-bench-instance:latest")

    logger.info(f"Handshake Complete: context_id={context_id[:20]} | repo={repo}")

    # Initialize Core with Phase 2 configurations
    llm = LLMClient()
    docker = DockerBridge(image_name=image_name)
    tester = TestEngine(docker)
    hyp_gen = HypothesisGenerator(llm)
    icl = ICLSpecialist()

    if not docker.start_container():
        return JSONResponse(content={"error": "Docker bridge failed to initialize"}, status_code=500)

    try:
        # Pipeline Execution (Stage 1 through 6)
        graph_builder = ASTGraphBuilder(repo_path="/workspace")
        repo_skeleton = graph_builder.build_repo_graph()
        
        tester.discover_and_run_baseline() # Stage 2 Baseline

        # Stage 1.5 & 2.5: Primer Synthesis
        hyp_group = await hyp_gen.generate_group(problem_statement, repo_skeleton)
        context_primer = f"{icl.get_injection(problem_statement, repo_skeleton)}\n" \
                         f"{icl.get_few_shot_examples()}\n" \
                         f"{hyp_gen.format_for_primer(hyp_group)}"

        # Stage 4: The Stateful Loop
        agent_loop = AgentLoop(llm, docker, tester)
        success, messages = await agent_loop.run_stage_4_bash_repl(problem_statement, context_primer)

        # Stage 6: The Mechanical Gate
        patch_content = ""
        if success:
            if await agent_loop.run_stage_6_qa_phase(messages):
                # Extract ground-truth patch from container state
                _, patch_content = docker.execute_command("git diff")

        # Final A2A Response formatting[cite: 2]
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": jsonrpc_id,
            "result": {
                "id": str(uuid.uuid4()),
                "contextId": context_id,
                "status": {"state": "completed"},
                "artifacts": [
                    {
                        "artifactId": str(uuid.uuid4()),
                        "name": "patch",
                        "parts": [{"kind": "text", "text": patch_content}]
                    }
                ]
            }
        })

    finally:
        docker.stop_container()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)