"""
AgentLoop — Phase 2 v4.2.2

Key design decisions:
  - System prompt uses generic placeholder REPO_ROOT (never /workspace)
  - repo_dir injected into every dynamic string at runtime, not at init time
  - _urgency_message is an instance method (not staticmethod) so it takes repo_dir
  - _append_to_notes and reasoning write both receive repo_dir as argument
  - MAX_TURNS and other constants are env-overridable
"""

import logging
import asyncio
import textwrap
from typing import Dict, List, Tuple

from .llm_client import LLMClient
from .docker_bridge import DockerBridge
from ..tools.test_engine import TestEngine

logger = logging.getLogger(__name__)

import os as _os

MAX_TURNS     = int(_os.getenv("MAX_TURNS",     "20"))
MAX_OBS_CHARS = int(_os.getenv("MAX_OBS_CHARS", "1500"))
CONTEXT_KEEP  = int(_os.getenv("CONTEXT_KEEP",  "8"))


class AgentLoop:

    def __init__(self, llm_client: LLMClient, docker_bridge: DockerBridge, test_engine: TestEngine):
        self.llm    = llm_client
        self.docker = docker_bridge
        self.tester = test_engine

        # System prompt uses REPO_ROOT as a readable placeholder.
        # The actual path is injected into the turn-1 user message at runtime.
        # REPO_ROOT placeholder is used instead — actual path injected at runtime.
        self.system_prompt = textwrap.dedent("""\
            You are Purple Agent, an expert software engineer in a stateful Bash REPL.
            You have a STRICT budget of 20 shell calls. Work efficiently.
            A top-tier engineer solves SWE-bench tasks in 8-12 calls.

            <protocol>
            Every response MUST use this exact XML structure:
            <thought>
            Step-by-step reasoning. Diagnose, locate, fix, verify.
            </thought>
            <action type="bash">
            single command or chained commands with &&
            </action>

            When your fix is verified:
            <action type="submit">Done</action>

            CRITICAL: NEVER wrap in markdown fences. No ```xml, ```bash, ```json.
            The raw XML tags must appear at the top level of your response.
            </protocol>

            <memory_rules>
            Your context window is pruned every few turns to control costs.
            TREAT REPO_ROOT/NOTES.txt AS YOUR PRIMARY EXTERNAL MEMORY.
            REPO_ROOT is the value shown in the ENVIRONMENT block at the top of your
            conversation. Use that exact path wherever you see REPO_ROOT in this prompt.
            The framework appends your bash output to NOTES.txt automatically each turn.
            If you feel lost or are repeating commands:
              cat REPO_ROOT/NOTES.txt
            </memory_rules>

            <tools>
            Three tools are injected into REPO_ROOT at startup — use them:

            1. FILE EDITOR (avoids sed whitespace pitfalls):
               python REPO_ROOT/edit_file.py "path/to/file" "exact old code" "new code"
               Old code must match CHARACTER-FOR-CHARACTER including whitespace.
               Always verify after: grep -n "new code" path/to/file

            2. AST SEARCH (finds definitions and call sites):
               python REPO_ROOT/ast_search.py "FunctionOrClassName"

            3. TEST RUNNER:
               bash REPO_ROOT/run_script.sh                      (full suite)
               pytest path/test.py::test_name -x --tb=short      (targeted, preferred)
               go test -run TestName ./pkg/...                    (Go targeted)
            </tools>

            <verification_rules>
            Before submitting:
            1. Run bash REPO_ROOT/run_script.sh or a targeted subset
            2. Confirm fix passes and no regressions are introduced
            3. Handle None/null, empty lists/dicts, boundary values
            A partial fix is better than no fix.
            </verification_rules>

            <efficiency_rules>
            - grep -n 'pattern' file | head -30   (targeted, not cat on large files)
            - sed -n '40,80p' file.py              (read a section by line range)
            - Run ONLY the specific failing test
            - Workflow: diagnose (1-5) -> edit (6) -> verify (7) -> submit (8)
            </efficiency_rules>
        """)

    def _bootstrap_workspace(self, repo_dir: str):
        """Inject helper scripts into repo_dir (auto-detected, never /workspace)."""

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

        ast_search = (
            "import sys, subprocess\n"
            "if len(sys.argv) < 2:\n"
            "    print('Usage: python ast_search.py <Name>')\n"
            "    sys.exit(1)\n"
            "target = sys.argv[1]\n"
            "exts = ['*.py','*.go','*.js','*.ts','*.tsx','*.rb','*.java','*.rs','*.c','*.cpp']\n"
            "includes = sum([['--include', e] for e in exts], [])\n"
            "pattern = rf'(def |func |class |function |fn |\\btype ).*\\b{target}\\b'\n"
            "r = subprocess.run(['grep','-rn','-E',pattern,'.']+includes,\n"
            "    capture_output=True, text=True, cwd='.')\n"
            "if r.stdout:\n"
            "    lines = r.stdout.strip().split('\\n')\n"
            "    print(f'Found {len(lines)} definition(s):\\n')\n"
            "    print('\\n'.join(lines[:40]))\n"
            "else:\n"
            "    r2 = subprocess.run(['grep','-rn','--include=*.py','--include=*.go',\n"
            "        '--include=*.js','--include=*.ts',target,'.']+[],\n"
            "        capture_output=True, text=True, cwd='.')\n"
            "    hits = r2.stdout.strip().split('\\n')[:20] if r2.stdout else []\n"
            "    print('\\n'.join(hits) if hits else 'Not found.')\n"
        )
        self.docker.execute_command(
            f"cat > {repo_dir}/ast_search.py << 'PYEOF'\n{ast_search}PYEOF"
        )

        test_cmd = self.tester.test_command or "echo 'No test runner found.'"
        self.docker.execute_command(
            f"printf '#!/bin/bash\\nset -e\\ncd {repo_dir}\\n{test_cmd}\\n' "
            f"> {repo_dir}/run_script.sh && chmod +x {repo_dir}/run_script.sh"
        )

        self.docker.execute_command(
            f"printf '### PURPLE AGENT NOTES ###\\n"
            f"- Repo root: {repo_dir}\\n"
            f"- Test cmd: {test_cmd[:80]}\\n"
            f"- Start of exploration.\\n' > {repo_dir}/NOTES.txt"
        )

    @staticmethod
    def _cap_observation(output: str) -> str:
        if len(output) <= MAX_OBS_CHARS:
            return output
        half = MAX_OBS_CHARS // 2
        return (
            output[:half]
            + f"\n... [TRUNCATED {len(output) - MAX_OBS_CHARS} chars] ...\n"
            + output[-half:]
        )

    def _append_to_notes(self, repo_dir: str, turn: int, command: str, output: str):
        summary = output.strip()[:200].replace("'", " ")
        note = f"Turn {turn}: $ {command[:80]} -> {summary}"
        self.docker.execute_command(
            f"printf '\\n{note}\\n' >> {repo_dir}/NOTES.txt 2>/dev/null || true",
            timeout=5,
        )

    @staticmethod
    def _prune_context(messages: List[Dict]) -> List[Dict]:
        if len(messages) <= 2 + CONTEXT_KEEP:
            return messages
        logger.info(
            "Context pruned: %d -> %d messages (kept system+task+last %d)",
            len(messages), 2 + CONTEXT_KEEP, CONTEXT_KEEP,
        )
        return messages[:2] + messages[-CONTEXT_KEEP:]

    def _urgency_message(self, turn: int, repo_dir: str) -> Dict | None:
        """Instance method — needs repo_dir for NOTES.txt path in message."""
        warn  = max(1, MAX_TURNS - 6)
        alert = max(1, MAX_TURNS - 2)
        remaining = MAX_TURNS - turn

        if turn == warn:
            return {
                "role": "user",
                "content": (
                    f"<observation status=\"SYSTEM\">"
                    f"WARNING: TURN {turn}/{MAX_TURNS}. {remaining} turns left. "
                    f"Run: cat {repo_dir}/NOTES.txt to review what you tried. "
                    f"Stop broad exploration. Locate the bug and edit now."
                    f"</observation>"
                ),
            }
        if turn == alert:
            return {
                "role": "user",
                "content": (
                    f"<observation status=\"SYSTEM\">"
                    f"CRITICAL: TURN {turn}/{MAX_TURNS}. {remaining} turns left. "
                    f"Use {repo_dir}/edit_file.py to make your edit and submit NOW. "
                    f"A partial fix beats no fix."
                    f"</observation>"
                ),
            }
        return None

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
                    f"REPO_ROOT = `{repo_dir}`\n"
                    f"All source files, tests, and injected tools "
                    f"(edit_file.py, ast_search.py, run_script.sh, NOTES.txt) "
                    f"are inside `{repo_dir}`.\n"
                    f"Replace REPO_ROOT with `{repo_dir}` everywhere.\n\n"
                    f"### TARGET ISSUE\n{issue_text}\n\n"
                    f"{context_primer}\n\n"
                    f"Begin: cd {repo_dir} && <your first diagnostic command>"
                ),
            },
        ]

        logger.info("Stage 4: starting %d-turn REPL (repo_dir=%s)", MAX_TURNS, repo_dir)

        for turn in range(1, MAX_TURNS + 1):
            logger.info("--- TURN %d/%d ---", turn, MAX_TURNS)

            if turn == MAX_TURNS:
                logger.warning("Turn budget exhausted. Force-extracting git diff.")
                _, diff = self.docker.execute_command(
                    f"cd {repo_dir} && git diff HEAD", timeout=20
                )
                logger.info(
                    "Force-submit: diff has %d chars", len(diff)
                ) if diff.strip() else logger.warning("Force-submit: diff is empty")
                return True, messages

            urg = self._urgency_message(turn, repo_dir)
            if urg:
                messages.append(urg)

            messages = self._prune_context(messages)

            try:
                raw = await self.llm.generate_step(messages)
            except Exception as e:
                logger.error("LLM error turn %d: %s", turn, str(e)[:150])
                messages.append({
                    "role": "user",
                    "content": f"<observation status=\"FAILED\">LLM error: {str(e)[:200]}</observation>",
                })
                continue

            messages.append({"role": "assistant", "content": raw})

            # Gemini reasoning -> NOTES.txt (side-channel, no context pollution)
            reasoning = getattr(self.llm, "_last_reasoning", "")
            if reasoning and len(reasoning) > 20:
                summary = reasoning[:400].replace("'", " ").replace('"', " ").replace("\n", " ")
                self.docker.execute_command(
                    f"printf '\\n[Turn {turn} thinking]: {summary}\\n' "
                    f">> {repo_dir}/NOTES.txt 2>/dev/null || true",
                    timeout=5,
                )

            try:
                thought, action_type, action_content = self.llm.parse_response(raw)
            except Exception as e:
                logger.error("Parse error turn %d: %s", turn, e)
                thought, action_type, action_content = (
                    "", "bash",
                    "echo 'Parse error. Use <action type=\"bash\">cmd</action>'"
                )

            logger.info("Turn %d | action=%s | content=%s", turn, action_type, action_content[:80])

            if not action_type:
                messages.append({
                    "role": "user",
                    "content": "<observation status=\"FAILED\">Missing action tag.</observation>",
                })
                continue

            if action_type == "submit":
                logger.info("Agent submitted on turn %d", turn)
                return True, messages

            if action_type == "bash":
                exit_code, output = self.docker.execute_command(action_content)
                output_capped = self._cap_observation(output)
                self._append_to_notes(repo_dir, turn, action_content, output)

                is_edit = any(kw in action_content for kw in
                              ["edit_file.py", "git apply", "tee ", "patch "])
                if turn % 4 == 0 or is_edit:
                    self.docker.execute_command(
                        f"cd {repo_dir} && git diff HEAD > /tmp/purple_patch.diff 2>/dev/null || true",
                        timeout=10,
                    )

                messages.append({"role": "user", "content": self.llm.format_observation(output_capped, exit_code)})
            else:
                messages.append({
                    "role": "user",
                    "content": f"<observation status=\"FAILED\">Unknown action '{action_type}'. Use bash or submit.</observation>",
                })

        return True, messages

    async def run_stage_6_qa_phase(self, messages: List[Dict], max_qa_retries: int = 2) -> bool:
        for attempt in range(1, max_qa_retries + 1):
            logger.info("QA attempt %d/%d", attempt, max_qa_retries)
            gate_passed, gate_msg = self.tester.verify_patch()
            if gate_passed:
                logger.info("QA gate passed on attempt %d", attempt)
                return True

            failing_tests = [
                l.strip() for l in gate_msg.splitlines()
                if l.strip().startswith("FAILED ") or l.strip().startswith("--- FAIL:")
            ]
            test_list = "\n".join(failing_tests[:5]) if failing_tests else gate_msg[:500]
            targeted_cmd = self.tester.test_command or "git diff HEAD"
            if failing_tests:
                targeted_cmd = (
                    "pytest " + " ".join(t.replace("FAILED ", "").split(" - ")[0] for t in failing_tests[:3]) + " -x --tb=short"
                    if failing_tests[0].startswith("FAILED") else
                    "go test -run '" + "|".join(t.replace("--- FAIL: ", "").split("(")[0] for t in failing_tests[:3]) + "' ./..."
                )

            messages.append({
                "role": "user",
                "content": (
                    f"GATE FAILED (attempt {attempt}/{max_qa_retries}).\n"
                    f"Failing tests:\n{test_list}\n\n"
                    f"Run targeted: `{targeted_cmd}`\n"
                    f"Fix and submit."
                ),
            })

            for _ in range(3):
                try:
                    raw = await self.llm.generate_step(self._prune_context(messages))
                except Exception as e:
                    logger.error("QA LLM error: %s", e)
                    break
                messages.append({"role": "assistant", "content": raw})
                try:
                    _, atype, acontent = self.llm.parse_response(raw)
                except Exception:
                    atype, acontent = "bash", "echo 'parse error'"
                if atype == "submit":
                    break
                if atype == "bash":
                    ec, out = self.docker.execute_command(acontent)
                    messages.append({"role": "user", "content": self.llm.format_observation(self._cap_observation(out), ec)})

        logger.error("QA gate exhausted after %d attempts", max_qa_retries)
        return False