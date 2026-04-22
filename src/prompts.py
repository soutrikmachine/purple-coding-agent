"""
Prompts
=======
All prompt templates live here to keep them easy to iterate on.

The model is trained (via GRPO) to output in structured XML:
    <thought>  reasoning about the problem          </thought>
    <command>  bash | debug | patch                 </command>
    <content>  the actual command / diff            </content>

This format lets the PRM score the <thought> TIR separately from execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import SWETask

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a world-class software engineer debugging a real GitHub issue.

You have access to an interactive shell in the project repository. You work
in three modes:

  bash   – Read-only exploration (grep, find, cat, python -c, git log, …)
  debug  – Temporary writes + execution (sed, echo >> file, python script.py)
  patch  – Submit the final unified diff that fixes the bug

ALWAYS output your response in this XML format:

<thought>
Step-by-step analysis of what you know and what to do next.
Reference specific file names, line numbers, and function names.
Verify your reasoning before acting.
</thought>
<command>bash|debug|patch</command>
<content>
<the actual shell command OR unified diff here>
</content>

RULES:
1. Start by understanding the repo structure with 'find' and 'git log'.
2. Read relevant source files with 'cat -n' to see line numbers.
3. Run failing tests with 'python -m pytest <test_path> -x' in bash mode first.
4. In debug mode, you may write temp files to verify your fix logic.
5. Only switch to patch mode when you are confident in the fix.
6. The patch must be a valid unified diff (git diff format).
7. Do NOT use 'rm', 'dd', or destructive commands.
8. Keep bash commands focused and short — prefer one action at a time.
"""

# ── User Prompts ──────────────────────────────────────────────────────────────


def build_user_prompt(task: SWETask) -> str:
    """Build the initial task description prompt."""
    lines = [
        "# Bug Report",
        "",
        task.problem_statement.strip(),
    ]

    if task.hints_text:
        lines += ["", "## Hints", task.hints_text.strip()]

    if task.fail_to_pass:
        lines += [
            "",
            "## Tests That Must Pass After Your Fix",
            *[f"  - {t}" for t in task.fail_to_pass],
        ]

    lines += [
        "",
        f"## Environment",
        f"  Working directory: {task.cwd}",
        f"  Python version: {task.python_version}",
        "",
        "Begin your exploration. Output your first action.",
    ]

    return "\n".join(lines)


def build_observation_prompt(obs: dict) -> str:
    """Build the observation message after an action was executed."""
    parts = []

    cwd = obs.get("cwd", "")
    if cwd:
        parts.append(f"[cwd: {cwd}]")

    stdout = obs.get("stdout", "").strip()
    stderr = obs.get("stderr", "").strip()

    if stdout:
        # Truncate very long outputs to fit context window
        if len(stdout) > 4000:
            stdout = stdout[:2000] + "\n…[truncated]…\n" + stdout[-500:]
        parts.append(f"stdout:\n```\n{stdout}\n```")

    if stderr:
        if len(stderr) > 1000:
            stderr = stderr[:500] + "\n…[truncated]…\n" + stderr[-200:]
        parts.append(f"stderr:\n```\n{stderr}\n```")

    if not stdout and not stderr:
        parts.append("(no output)")

    return "\n".join(parts) + "\n\nWhat is your next action?"


def build_grpo_reward_prompt(trajectory: list[dict], task: SWETask) -> str:
    """
    Prompt used during GRPO training to evaluate a full trajectory.
    Returns a prompt that asks the model to self-critique.
    """
    history_lines = []
    for i, step in enumerate(trajectory, 1):
        action = step.get("action", {})
        obs = step.get("observation", {})
        history_lines.append(
            f"Step {i}: [{action.get('action')}] {action.get('content', '')[:100]}"
        )
        if obs.get("stdout"):
            history_lines.append(f"  → {obs['stdout'][:100]}")

    history_text = "\n".join(history_lines)

    return f"""\
Review this debugging trajectory for the following bug:

{task.problem_statement[:500]}

Trajectory:
{history_text}

Evaluate:
1. Did the agent correctly identify the root cause? (yes/no/partial)
2. Is the logic in the <thought> blocks sound?
3. Are the bash commands efficient and targeted?
4. Is the final patch correct and minimal?

Output a score from 0.0 to 1.0 and brief justification.
"""
