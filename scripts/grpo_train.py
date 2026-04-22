"""
GRPO Training Script (Kaggle 2xT4)
====================================
Week 2, Hours 11–15: Fine-tune DeepSeek-Coder-V2-Lite-Instruct on gold
trajectories collected in Week 1 using Group Relative Policy Optimization.

Run this in your Kaggle notebook:
    !python scripts/grpo_train.py --data gold_trajectories.jsonl --output /kaggle/working/grpo_model

Requirements:
    unsloth, trl>=0.9, datasets, torch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("grpo_train")


# ── Reward Functions ──────────────────────────────────────────────────────────

def reward_format(completions: list[str], **kwargs) -> list[float]:
    """
    R_f: Reward valid XML structure with <thought>, <command>, <content>.
    """
    import re
    rewards = []
    for c in completions:
        score = 0.0
        if "<thought>" in c and "</thought>" in c:
            score += 0.33
        if "<command>" in c and "</command>" in c:
            cmd = re.search(r"<command>(.*?)</command>", c, re.DOTALL)
            if cmd and cmd.group(1).strip() in ("bash", "debug", "patch"):
                score += 0.33
        if "<content>" in c and "</content>" in c:
            content = re.search(r"<content>(.*?)</content>", c, re.DOTALL)
            if content and len(content.group(1).strip()) > 5:
                score += 0.34
        rewards.append(score)
    return rewards


def reward_logic(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    R_l: Reward thought blocks that reference specific file paths / functions.
    """
    import re
    rewards = []
    for completion, prompt in zip(completions, prompts):
        thought_match = re.search(r"<thought>(.*?)</thought>", completion, re.DOTALL)
        if not thought_match:
            rewards.append(0.0)
            continue
        thought = thought_match.group(1)

        # Extract file paths from thought
        files_in_thought = re.findall(r'[\w/.-]+\.py(?::\d+)?', thought)

        # Check if thought mentions files found in the codebase (proxy: prompt mentions them)
        files_in_prompt = re.findall(r'[\w/.-]+\.py(?::\d+)?', prompt)
        prompt_file_set = {f.split("/")[-1] for f in files_in_prompt}
        thought_file_set = {f.split("/")[-1] for f in files_in_thought}

        overlap = len(prompt_file_set & thought_file_set)
        score = min(overlap / max(len(prompt_file_set), 1), 1.0) if prompt_file_set else 0.5

        # Bonus for algorithmic consistency check
        if "binary search" in thought.lower() and "linear" in thought.lower():
            score -= 0.2  # caught an inconsistency

        rewards.append(max(0.0, score))
    return rewards


def reward_patch(completions: list[str], ground_truth: list[str], **kwargs) -> list[float]:
    """
    R_e: Reward patches that match gold trajectory outcomes.
    Simplified: check if the patch command is present and has a diff header.
    """
    import re
    rewards = []
    for completion, gold in zip(completions, ground_truth):
        cmd_match = re.search(r"<command>(.*?)</command>", completion, re.DOTALL)
        content_match = re.search(r"<content>(.*?)</content>", completion, re.DOTALL)

        if not cmd_match or not content_match:
            rewards.append(0.0)
            continue

        cmd = cmd_match.group(1).strip()
        content = content_match.group(1).strip()

        if cmd == "patch":
            has_diff = "diff --git" in content or "--- a/" in content
            has_hunk = "@@" in content
            has_adds = re.search(r'^\+', content, re.MULTILINE) is not None
            score = (0.33 * has_diff) + (0.33 * has_hunk) + (0.34 * has_adds)
        elif cmd in ("bash", "debug"):
            # Non-patch steps: reward for non-empty, reasonable commands
            score = 0.5 if content else 0.0
        else:
            score = 0.0

        rewards.append(float(score))
    return rewards


# ── Dataset Builder ───────────────────────────────────────────────────────────

def load_trajectories(path: str) -> list[dict]:
    """Load JSONL file of gold trajectories collected in Week 1."""
    trajectories = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    logger.info("Loaded %d trajectories from %s", len(trajectories), path)
    return trajectories


def trajectory_to_examples(traj: dict) -> list[dict]:
    """
    Convert a single trajectory into training examples.
    Each step becomes one (prompt, completion, ground_truth) example.
    """
    from prompts import SYSTEM_PROMPT, build_user_prompt, build_observation_prompt
    from agent import SWETask

    task_data = traj.get("task", {})
    task = SWETask(
        problem_statement=task_data.get("problem_statement", ""),
        cwd=task_data.get("cwd", "/workspace/repo"),
        hints_text=task_data.get("hints_text", ""),
        fail_to_pass=task_data.get("fail_to_pass", []),
    )

    steps = traj.get("steps", [])
    examples = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": build_user_prompt(task)})

    for step in steps:
        action = step.get("action", {})
        obs = step.get("observation", {})

        # The ground-truth completion is the formatted action XML
        completion = (
            f"<thought>\n{step.get('thought', 'Analysing the problem.')}\n</thought>\n"
            f"<command>{action.get('action', 'bash')}</command>\n"
            f"<content>\n{action.get('content', '')}\n</content>"
        )

        examples.append({
            "messages": list(messages),
            "ground_truth": completion,
            "task_problem": task.problem_statement[:200],
        })

        # Advance conversation
        messages.append({"role": "assistant", "content": completion})
        if obs:
            messages.append({"role": "user", "content": build_observation_prompt(obs)})

    return examples


# ── Main Training Loop ────────────────────────────────────────────────────────

def train(args):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    logger.info("Loading Unsloth + model…")
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_len,
            load_in_4bit=True,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        logger.info("Model loaded with Unsloth LoRA ✓")
    except ImportError:
        logger.error("Unsloth not installed. Run: pip install unsloth")
        raise

    # Load and build dataset
    trajectories = load_trajectories(args.data)
    all_examples = []
    for traj in trajectories:
        all_examples.extend(trajectory_to_examples(traj))

    logger.info("Total training examples: %d", len(all_examples))

    from datasets import Dataset
    ds = Dataset.from_list(all_examples)

    # Apply chat template to messages
    def format_prompt(examples):
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
        return {"prompt": texts}

    ds = ds.map(format_prompt, batched=True, remove_columns=["messages"])

    # GRPO config
    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=10,
        save_steps=50,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        # GRPO-specific
        num_generations=args.num_generations,
        max_new_tokens=1024,
        temperature=0.7,
        report_to="none" if not args.wandb else "wandb",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=ds,
        reward_funcs=[reward_format, reward_logic, reward_patch],
        tokenizer=tokenizer,
    )

    logger.info("🚀 Starting GRPO training for %d epochs…", args.epochs)
    trainer.train()

    # Save final model
    output_path = Path(args.output)
    model.save_pretrained_merged(
        str(output_path / "final"),
        tokenizer,
        save_method="merged_16bit",
    )
    logger.info("✅ Model saved to %s/final", args.output)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GRPO fine-tuning for Purple Agent")
    p.add_argument("--data", required=True, help="Path to gold_trajectories.jsonl")
    p.add_argument("--output", default="/kaggle/working/grpo_model")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
