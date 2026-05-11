"""
AgentLoop — Phase 2 v4.2.1

Key fixes vs. submitted version:
  - max_turns: 50 → 15  (50 turns = 695s > 300s gateway = 100% 504)
  - Context window: keeps system + task + rolling last-8 messages
    (prevents token explosion while keeping recent state visible)
  - Auto-NOTES.txt: appended by the loop after EVERY bash exec
    (agent can't forget what it tried — the framework writes it)
  - Urgency escalation at turns 10, 13, 15 — prevents "exploration forever"
  - Force-submit at turn 15: extract git diff, terminate, no more waiting
  - Observation cap: 1500 chars (head + tail), up from 1000 (was losing errors)
"""

import logging
import asyncio
import re
import textwrap
from typing import Dict, List, Tuple

from .llm_client import LLMClient
from .docker_bridge import DockerBridge
from ..tools.test_engine import TestEngine

logger = logging.getLogger(__name__)

import os as _os

# ── Tunable constants (all overridable via env vars) ───────────────────────────
MAX_TURNS     = int(_os.getenv("MAX_TURNS",     "20"))   # turns before force-submit
MAX_OBS_CHARS = int(_os.getenv("MAX_OBS_CHARS", "1500")) # observation cap (chars)
CONTEXT_KEEP  = int(_os.getenv("CONTEXT_KEEP",  "8"))    # rolling message window

# ── Urgency thresholds (derived from MAX_TURNS) ────────────────────────────────
URGENCY_WARN  = max(1, MAX_TURNS - 6)   # 6 turns before end: "running out"
URGENCY_ALERT = max(1, MAX_TURNS - 2)   # 2 turns before end: "submit NOW"
FORCE_SUBMIT  = MAX_TURNS               # on this turn, extract diff and stop


class AgentLoop:
    """
    15-turn stateful bash REPL.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        docker_bridge: DockerBridge,
        test_engine: TestEngine,
    ):
        self.llm    = llm_client
        self.docker = docker_bridge
        self.tester = test_engine

        self.system_prompt = textwrap.dedent("""\
            You are Purple Agent, an expert software engineer in a stateful Bash REPL.
            The repository root is shown in the ENVIRONMENT block of the first message. You have a STRICT budget of 20 shell calls.
            A top-tier engineer solves SWE-bench tasks in 8-12 calls. Work efficiently.

            <protocol>
            Every response MUST use this exact XML structure — no exceptions:
            <thought>
            Step-by-step reasoning. Diagnose → locate → fix → verify.
            </thought>
            <action type="bash">
            single command or chained commands with &&
            </action>

            When your fix is verified and ready, terminate with:
            <action type="submit">Done</action>

            CRITICAL FORMATTING RULES:
            - NEVER wrap your response in markdown fences (no ```xml, ```bash, ```json)
            - NEVER add any text before <thought> or after </action>
            - The raw XML tags must appear at the top level of your response
            - Violating this causes the framework to fail silently and wastes a turn
            </protocol>

            <memory_rules>
            Your context window is pruned to keep costs manageable.
            TREAT {REPO_ROOT}/NOTES.txt AS YOUR PRIMARY EXTERNAL MEMORY.
            (REPO_ROOT is shown in the ENVIRONMENT block — use the actual path, not /workspace)
            The framework automatically appends your bash outputs there,
            but you MUST prefix your thought with a summary line like:
              "Step N: [what I found / what I changed]"
            If you feel lost, your FIRST action must be:
              cat {repo_dir}/NOTES.txt
            </memory_rules>

            <tools>
            Two injected tools live at /workspace — use them:

            1. FILE EDITOR (avoids sed pitfalls):
               python {repo_dir}/edit_file.py "path/to/file" "exact old code" "new code"
               'exact old code' must match CHARACTER-FOR-CHARACTER. Verify after:
               grep -n "new code" path/to/file

            2. AST SEARCH (finds definitions and call sites instantly):
               python {repo_dir}/ast_search.py "FunctionOrClassName"
               Use this before blind grep when looking for where something is defined.

            3. TEST RUNNER:
               bash {repo_dir}/run_script.sh           ← full suite
               pytest path/test.py::test_name -x --tb=short  ← targeted (preferred)
               go test -run TestName ./pkg/...          ← Go targeted
            </tools>

            <verification_rules>
            Before submitting:
            1. Run bash {repo_dir}/run_script.sh (or a targeted subset)
            2. Confirm fix passes and no regressions introduced
            3. Handle None/null, empty lists/dicts, boundary values
            A partial working fix is better than nothing — submit when stuck.
            </verification_rules>

            <efficiency_rules>
            - grep -n 'pattern' file | head -30  (not cat on large files)
            - sed -n '40,80p' file.py  (read section by line range)
            - Run ONLY the failing test, not the full suite
            - After finding bug at turn 5: edit turn 6, verify turn 7, submit turn 8
            </efficiency_rules>
        """)

    # ── Workspace bootstrap ────────────────────────────────────────────────────

    def _bootstrap_workspace(self, repo_dir: str):
        """
        Inject three helper scripts into the REPO directory (not hardcoded /workspace).
        The repo_dir is auto-detected by DockerBridge._detect_repo_dir() and varies
        per SWE-bench image: /testbed, /app, /repo, etc.

        Tools written to {repo_dir}/:
          - edit_file.py  : safe file editor (avoids sed pitfalls)
          - ast_search.py : grep-based function/class locator (no tree-sitter needed)
          - run_script.sh : discovered test command wrapper
          - NOTES.txt     : agent scratchpad (framework writes here each turn)
        """
        # ── 1. edit_file.py ───────────────────────────────────────────────────
        editor = (
            "import sys\n"
            "f, old, new = sys.argv[1], sys.argv[2], sys.argv[3]\n"
            "c = open(f).read()\n"
            "if old in c:\n"
            "    open(f,'w').write(c.replace(old,new,1))\n"
            "    print(f'SUCCESS: replaced in {f}')\n"
            "else:\n"
            "    print(f'ERROR: old_text not found in {f}. Check whitespace!')\n"
        )
        self.docker.execute_command(
            f"cat > {repo_dir}/edit_file.py << 'PYEOF'\n{editor}PYEOF"
        )

        # ── 2. ast_search.py (grep-based, no tree-sitter dependency) ─────────
        # Uses grep to find function/class definitions across all languages.
        # Much faster than tree-sitter for the agent's lookup use case.
        ast_search = (
            "import sys, subprocess\n"
            "if len(sys.argv) < 2:\n"
            "    print('Usage: python {repo_dir}/ast_search.py <Name>')\n"
            "    sys.exit(1)\n"
            "target = sys.argv[1]\n"
            "exts = ['*.py','*.go','*.js','*.ts','*.tsx','*.rb','*.java','*.rs','*.c','*.cpp']\n"
            "includes = sum([['--include', e] for e in exts], [])\n"
            "# Pattern covers: def X, func X, class X, function X, fn X, method X\n"
            "pattern = rf'(def |func |class |function |fn |\btype ).*\\b{target}\\b'\n"
            "r = subprocess.run(['grep','-rn','-E',pattern,'.']+includes,\n"
            "    capture_output=True, text=True, cwd='.')\n"
            "if r.stdout:\n"
            "    lines = r.stdout.strip().split('\\n')\n"
            "    print(f'Found {len(lines)} definition(s) of \'{target}\':\\n')\n"
            "    print('\\n'.join(lines[:40]))\n"
            "else:\n"
            "    # Fallback: any reference to the name\n"
            "    r2 = subprocess.run(['grep','-rn','--include=*.py','--include=*.go',\n"
            "        '--include=*.js','--include=*.ts',target,'.']+[],\n"
            "        capture_output=True, text=True, cwd='.')\n"
            "    hits = r2.stdout.strip().split('\\n')[:20] if r2.stdout else []\n"
            "    if hits:\n"
            "        print(f'No definition found. References to \'{target}\':\\n')\n"
            "        print('\\n'.join(hits))\n"
            "    else:\n"
            "        print(f'\'{target}\' not found anywhere in the repo.')\n"
        )
        self.docker.execute_command(
            f"cat > {repo_dir}/ast_search.py << 'PYEOF'\n{ast_search}PYEOF"
        )

        # ── 3. run_script.sh (wraps the discovered test command) ─────────────
        test_cmd = self.tester.test_command or "echo 'No test runner discovered. Run tests manually.'"
        self.docker.execute_command(
            f"printf '#!/bin/bash\\nset -e\\ncd {repo_dir}\\n{test_cmd}\\n' "
            "> {repo_dir}/run_script.sh && chmod +x {repo_dir}/run_script.sh"
        )

        # ── 4. Initialise NOTES.txt scratchpad ────────────────────────────────
        self.docker.execute_command(
            "printf '### PURPLE AGENT NOTES ###\\n- Start of exploration.\\n"
            "- Test runner: " + test_cmd[:80] + "\\n' > {repo_dir}/NOTES.txt"
        )

    # ── Observation handling ───────────────────────────────────────────────────

    @staticmethod
    def _cap_observation(output: str) -> str:
        """Keep head + tail of output, max MAX_OBS_CHARS total."""
        if len(output) <= MAX_OBS_CHARS:
            return output
        half = MAX_OBS_CHARS // 2
        removed = len(output) - MAX_OBS_CHARS
        return (
            output[:half]
            + f"\n... [TRUNCATED {removed} chars] ...\n"
            + output[-half:]
        )

    def _append_to_notes(self, repo_dir: str, turn: int, command: str, output: str):
        """Auto-write a summary line to NOTES.txt after every bash exec."""
        # First 200 chars of output — enough to capture errors or key values
        summary = output.strip()[:200].replace("'", "'\\''")
        note = f"Turn {turn}: $ {command[:80]}\\n  → {summary}"
        self.docker.execute_command(
            f"printf '\\n{note}\\n' >> {repo_dir}/NOTES.txt 2>/dev/null || true"
        )

    # ── Context window management ──────────────────────────────────────────────

    @staticmethod
    def _prune_context(messages: List[Dict]) -> List[Dict]:
        """
        Keep: messages[0] (system) + messages[1] (task/primer) + last CONTEXT_KEEP.

        Why this works:
        - System prompt is always present (instructions never forgotten)
        - Task message is always present (problem statement never forgotten)
        - Last 8 messages ≈ 4 turns of thought+observation (recent memory)
        - NOTES.txt provides long-term memory via bash (framework-written)
        """
        if len(messages) <= 2 + CONTEXT_KEEP:
            return messages
        logger.info(
            "Context pruned: %d → %d messages (kept system+task+last %d)",
            len(messages), 2 + CONTEXT_KEEP, CONTEXT_KEEP,
        )
        return messages[:2] + messages[-CONTEXT_KEEP:]

    # ── Urgency injections ─────────────────────────────────────────────────────

    @staticmethod
    def _urgency_message(turn: int) -> Dict | None:
        remaining = MAX_TURNS - turn
        if turn == URGENCY_WARN:
            return {
                "role": "user",
                "content": (
                    f"<observation status=\"SYSTEM\">"
                    f"⚠️  TURN {turn}/{MAX_TURNS}. {remaining} turns left. "
                    f"If you haven't found the bug yet, check /workspace/NOTES.txt "
                    f"then grep more specifically. Stop broad exploration."
                    f"</observation>"
                ),
            }
        if turn == URGENCY_ALERT:
            return {
                "role": "user",
                "content": (
                    f"<observation status=\"SYSTEM\">"
                    f"🚨 CRITICAL: TURN {turn}/{MAX_TURNS}. Only {remaining} turns left. "
                    f"You MUST make your edit NOW and submit. "
                    f"If you have a partial fix, submit it — a partial fix is better than no patch."
                    f"</observation>"
                ),
            }
        return None

    # ── Stage 4: Main REPL loop ────────────────────────────────────────────────

    async def run_stage_4_bash_repl(
        self,
        issue_text: str,
        context_primer: str,
    ) -> Tuple[bool, List[Dict]]:

        repo_dir = self.docker.repo_dir
        self._bootstrap_workspace(repo_dir)

        messages: List[Dict] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"### ENVIRONMENT\n"
                    f"Repository root: `{repo_dir}`\n"
                    f"All source files, tests, and your tools (edit_file.py, "
                    f"ast_search.py, run_script.sh, NOTES.txt) are inside `{repo_dir}`.\n"
                    f"When the system prompt says '/workspace', substitute `{repo_dir}`.\n\n"
                    f"### TARGET ISSUE\n{issue_text}\n\n"
                    f"{context_primer}\n\n"
                    f"Start in `{repo_dir}`. Begin by verifying the issue exists, "
                    "then locate and fix it."
                ),
            },
        ]

        logger.info("Stage 4: starting %d-turn REPL", MAX_TURNS)

        for turn in range(1, MAX_TURNS + 1):
            logger.info("--- TURN %d/%d ---", turn, MAX_TURNS)

            # Force-submit on final turn: extract diff and terminate
            if turn == FORCE_SUBMIT:
                logger.warning("Turn budget exhausted. Force-extracting git diff.")
                _, diff = self.docker.execute_command(f"cd {repo_dir} && git diff HEAD", timeout=20)
                if diff.strip():
                    logger.info("Force-submit: diff has %d chars", len(diff))
                else:
                    logger.warning("Force-submit: git diff is empty")
                return True, messages

            # Inject urgency message if at threshold
            urg = self._urgency_message(turn)
            if urg:
                messages.append(urg)

            # Prune context before each LLM call
            messages = self._prune_context(messages)

            # LLM call
            try:
                raw = await self.llm.generate_step(messages)
            except Exception as e:
                logger.error("LLM error on turn %d: %s", turn, str(e)[:150])
                messages.append({
                    "role": "user",
                    "content": f"<observation status=\"FAILED\">LLM error: {str(e)[:200]}. Retrying.</observation>",
                })
                continue

            messages.append({"role": "assistant", "content": raw})

            # For Gemini: write thinking summary to NOTES.txt as long-term reasoning memory.
            # This does NOT go into the message context (avoids token blowup).
            reasoning = getattr(self.llm, "_last_reasoning", "")
            if reasoning and len(reasoning) > 20:
                # Write first 400 chars of thinking — enough to capture the key decision
                summary = reasoning[:400].replace("'", " ").replace('"', " ").replace("\n", " ")
                self.docker.execute_command(
                    f"printf '\n[Turn {turn} thinking]: {summary}\n' "
                    ">> {repo_dir}/NOTES.txt 2>/dev/null || true",
                    timeout=5,
                )

            # Parse response
            try:
                thought, action_type, action_content = self.llm.parse_response(raw)
            except Exception as e:
                logger.error("Parse error turn %d: %s", turn, e)
                thought, action_type, action_content = (
                    "", "bash",
                    "echo 'Parse error — use exact XML: <action type=\"bash\">cmd</action>'"
                )

            logger.info("Turn %d | action=%s | content=%s", turn, action_type, action_content[:80])

            if not action_type:
                messages.append({
                    "role": "user",
                    "content": "<observation status=\"FAILED\">Missing <action> tag. Provide one.</observation>",
                })
                continue

            # Submit
            if action_type == "submit":
                logger.info("Agent submitted on turn %d", turn)
                return True, messages

            # Bash
            if action_type == "bash":
                exit_code, output = self.docker.execute_command(action_content)
                output_capped = self._cap_observation(output)

                # Auto-write to NOTES.txt (framework-level memory, not agent-level)
                self._append_to_notes(repo_dir, turn, action_content, output)

                # Periodic diff snapshot every 4 turns (or after any edit)
                # This ensures the global timeout handler in server.py can always
                # recover the best-effort patch even if we time out mid-loop.
                is_edit = any(kw in action_content for kw in
                              ["edit_file.py", "git apply", ">", ">>", "tee ", "patch "])
                if turn % 4 == 0 or is_edit:
                    self.docker.execute_command(
                        f"cd {repo_dir} && git diff HEAD > /tmp/purple_patch.diff 2>/dev/null || true",
                        timeout=10,
                    )

                obs = self.llm.format_observation(output_capped, exit_code)
                messages.append({"role": "user", "content": obs})
            else:
                messages.append({
                    "role": "user",
                    "content": (
                        f"<observation status=\"FAILED\">"
                        f"Unknown action '{action_type}'. Use 'bash' or 'submit'."
                        f"</observation>"
                    ),
                })

        return True, messages

    # ── Stage 6: QA fix phase ──────────────────────────────────────────────────

    async def run_stage_6_qa_phase(
        self,
        messages: List[Dict],
        max_qa_retries: int = 2,
    ) -> bool:
        """
        Targeted QA micro-loop: up to 2 retries × 3 sub-turns each.
        Injects specific failing test names so the agent runs targeted commands.
        """
        for attempt in range(1, max_qa_retries + 1):
            logger.info("QA attempt %d/%d", attempt, max_qa_retries)

            gate_passed, gate_msg = self.tester.verify_patch()
            if gate_passed:
                logger.info("QA gate passed on attempt %d", attempt)
                return True

            # Build targeted feedback
            # Extract specific test IDs from the gate message
            failing_tests = []
            for line in gate_msg.splitlines():
                stripped = line.strip()
                if stripped.startswith("FAILED ") or stripped.startswith("--- FAIL:"):
                    failing_tests.append(stripped)

            if failing_tests:
                test_list = "\n".join(failing_tests[:5])
                targeted_cmd = (
                    "pytest " + " ".join(
                        t.replace("FAILED ", "").split(" - ")[0]
                        for t in failing_tests[:3]
                    ) + " -x --tb=short"
                    if failing_tests[0].startswith("FAILED") else
                    "go test -run '" + "|".join(
                        t.replace("--- FAIL: ", "").split("(")[0]
                        for t in failing_tests[:3]
                    ) + "' ./..."
                )
            else:
                test_list = gate_msg[:500]
                targeted_cmd = self.tester.test_command or "git diff HEAD"

            messages.append({
                "role": "user",
                "content": (
                    f"GATE FAILED (attempt {attempt}/{max_qa_retries}).\n"
                    f"These specific tests are failing:\n{test_list}\n\n"
                    f"Run ONLY these targeted tests (faster feedback):\n"
                    f"  `{targeted_cmd}`\n\n"
                    f"Fix the failing tests with edit_file.py, then submit."
                ),
            })

            # 3-turn repair burst
            for sub_turn in range(3):
                try:
                    raw = await self.llm.generate_step(self._prune_context(messages))
                except Exception as e:
                    logger.error("QA LLM error: %s", e)
                    break

                messages.append({"role": "assistant", "content": raw})

                try:
                    _, atype, acontent = self.llm.parse_response(raw)
                except Exception:
                    atype, acontent = "bash", "echo 'Parse error'"

                if atype == "submit":
                    break

                if atype == "bash":
                    ec, out = self.docker.execute_command(acontent)
                    messages.append({
                        "role": "user",
                        "content": self.llm.format_observation(
                            self._cap_observation(out), ec
                        ),
                    })

        logger.error("QA gate exhausted after %d attempts", max_qa_retries)
        return False