import os
import json
import uuid
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Absolute imports are safer for the /app root context
from src.core.llm_client import LLMClient
from src.core.docker_bridge import DockerBridge
from src.core.agent_loop import AgentLoop
from src.tools.ast_graph import ASTGraphBuilder
from src.tools.hypotheses import HypothesisGenerator
from src.tools.test_engine import TestEngine
from src.prompts.icl_specialist import ICLSpecialist

# ── Logging & Config ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("purple_agent")

PORT = int(os.getenv("PORT", "9022"))

app = FastAPI(title="Purple Agent Phase 2")

# ==============================================================================
# A2A HANDSHAKE & HEALTH (Critical for "Ready" Status) ───────────────
# ==============================================================================

@app.get("/health")
async def health():
    """Standard health check for the Amber Router."""
    return {"status": "ok"}

@app.get("/.well-known/agent.json")
async def get_card():
    """Agent card required by the A2A protocol[cite: 1]."""
    return {
        "name": "Purple Agent",
        "description": "Phase 2 Execution Agent (PhoenixSmartLite)",
        "skills": ["coding", "testing", "bash", "git"]
    }

# ==============================================================================
# UTILITIES ─────────────────────────────────────────────────────────────
# ==============================================================================

def _extract_task_and_context(body: dict) -> tuple[dict, str]:
    """Parse A2A JSON-RPC envelope used by the Green Agent."""
    context_id = ""
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

            if kind == "data":
                data = part.get("data", {})
                if isinstance(data, dict) and "problem_statement" in data:
                    return data, context_id
            
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
    
    task_data, context_id = _extract_task_and_context(body)
    if not context_id:
        context_id = str(uuid.uuid4())
        
    problem_statement = task_data.get("problem_statement", "")
    repo = task_data.get("repo", "unknown/repo")
    image_name = task_data.get("container_image", "swe-bench-instance:latest")

    logger.info(f"Handshake Complete: context_id={context_id[:20]} | repo={repo}")

    llm = LLMClient()
    docker = DockerBridge(image_name=image_name)
    tester = TestEngine(docker)
    hyp_gen = HypothesisGenerator(llm)
    icl = ICLSpecialist()

    if not docker.start_container():
        return JSONResponse(content={"error": "Docker bridge failed to initialize"}, status_code=500)

    try:
        # Stage 1: Context Priming
        graph_builder = ASTGraphBuilder(repo_path="/workspace")
        repo_skeleton = graph_builder.build_repo_graph()
        
        # Stage 2: Baseline Execution
        tester.discover_and_run_baseline() 

        # Stage 3: Hypothesis Synthesis
        hyp_group = await hyp_gen.generate_group(problem_statement, repo_skeleton)
        context_primer = f"{icl.get_injection(problem_statement, repo_skeleton)}\n" \
                         f"{icl.get_few_shot_examples()}\n" \
                         f"{hyp_gen.format_for_primer(hyp_group)}"

        # Stage 4: The Stateful REPL Loop
        agent_loop = AgentLoop(llm, docker, tester)
        success, messages = await agent_loop.run_stage_4_bash_repl(problem_statement, context_primer)

        # Stage 5 & 6: Verification and Patch Extraction
        patch_content = ""
        if success:
            if await agent_loop.run_stage_6_qa_phase(messages):
                _, patch_content = docker.execute_command("git diff")

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