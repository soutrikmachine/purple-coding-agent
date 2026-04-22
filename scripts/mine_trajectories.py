"""
Trajectory Miner (Week 1, Hours 4–8)
======================================
Runs the base DeepSeek model against SWE-bench tasks to collect
"gold trajectories" where the model correctly finds the bug.

Trajectories are saved as JSONL for GRPO training.

Usage (Kaggle):
    !python scripts/mine_trajectories.py \
        --tasks 60 \
        --output /kaggle/working/gold_trajectories.jsonl \
        --model_url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("miner")


def load_swebench_tasks(split: str = "verified", max_tasks: int = 60) -> list[dict]:
    """Load SWE-bench tasks from HuggingFace datasets."""
    logger.info("Loading SWE-bench %s (max %d tasks)…", split, max_tasks)
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        tasks = list(ds)[:max_tasks]
        logger.info("Loaded %d tasks", len(tasks))
        return tasks
    except Exception as e:
        logger.error("Could not load SWE-bench: %s", e)
        raise


def run_agent_on_task(task: dict, model_url: str, max_turns: int = 10) -> dict | None:
    """
    Simulate running the agent against a SWE-bench task WITHOUT the full green agent.
    This is a simplified offline collection — we run the LLM and record its outputs.

    In production, this would call the actual green agent A2A endpoint.
    For mining, we just record the model's reasoning trajectory.
    """
    from llm_client import LLMClient
    from prompts import SYSTEM_PROMPT, build_user_prompt
    from agent import SWETask

    swe_task = SWETask(
        problem_statement=task.get("problem_statement", ""),
        cwd="/workspace/repo",
        hints_text=task.get("hints_text", ""),
        fail_to_pass=task.get("FAIL_TO_PASS", []) or [],
        pass_to_pass=task.get("PASS_TO_PASS", []) or [],
        repo=task.get("repo", ""),
    )

    llm = LLMClient(base_url=model_url)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(swe_task)},
    ]

    steps = []
    found_bug = False

    for turn in range(max_turns):
        try:
            raw = llm.complete(messages, temperature=0.7, max_tokens=1024)
        except Exception as e:
            logger.warning("LLM error on task %s turn %d: %s", task.get("instance_id"), turn, e)
            break

        # Parse action
        import re
        thought = _extract(raw, "thought") or ""
        command = (_extract(raw, "command") or "bash").strip().lower()
        content = _extract(raw, "content") or ""

        step = {
            "thought": thought,
            "action": {"action": command, "content": content},
            "observation": {},  # empty — no real execution
        }
        steps.append(step)
        messages.append({"role": "assistant", "content": raw})

        # Heuristic: check if model found the bug in its thought
        if any(kw in thought.lower() for kw in ["root cause", "the bug is", "found it", "line"]):
            found_bug = True

        if command == "patch" and content:
            found_bug = True
            break

        # Add dummy observation to continue conversation
        messages.append({"role": "user", "content": "(command executed)\n\nWhat is your next action?"})

    if not steps:
        return None

    return {
        "instance_id": task.get("instance_id", ""),
        "repo": task.get("repo", ""),
        "found_bug": found_bug,
        "task": {
            "problem_statement": swe_task.problem_statement,
            "cwd": swe_task.cwd,
            "hints_text": swe_task.hints_text,
            "fail_to_pass": swe_task.fail_to_pass,
        },
        "steps": steps,
        "num_turns": len(steps),
    }


def _extract(text: str, tag: str) -> str | None:
    import re
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def mine(args):
    tasks = load_swebench_tasks(max_tasks=args.tasks)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gold_count = 0
    total_count = 0

    with open(output_path, "a") as f:
        for i, task in enumerate(tasks):
            iid = task.get("instance_id", f"task_{i}")
            logger.info("[%d/%d] Mining %s…", i + 1, len(tasks), iid)

            result = run_agent_on_task(task, args.model_url, max_turns=args.max_turns)
            total_count += 1

            if result is None:
                continue

            # Save all trajectories (filter for gold during training)
            f.write(json.dumps(result) + "\n")
            f.flush()

            if result.get("found_bug"):
                gold_count += 1
                logger.info("  ✅ Bug found (%d gold so far)", gold_count)
            else:
                logger.info("  ❌ Bug not found")

            # Small delay to avoid overloading vLLM
            time.sleep(0.5)

    logger.info("Mining complete: %d/%d trajectories with bug found", gold_count, total_count)
    logger.info("Output: %s", output_path)


def parse_args():
    p = argparse.ArgumentParser(description="SWE-bench trajectory miner")
    p.add_argument("--tasks", type=int, default=60, help="Number of tasks to attempt")
    p.add_argument("--output", default="/kaggle/working/gold_trajectories.jsonl")
    p.add_argument("--model_url", default="http://localhost:8000")
    p.add_argument("--max_turns", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mine(args)
