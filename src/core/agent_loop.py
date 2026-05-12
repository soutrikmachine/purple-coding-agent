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
import time
from typing import Dict, List, Tuple

from .llm_client import LLMClient
from .docker_bridge import DockerBridge
from ..tools.test_engine import TestEngine

logger = logging.getLogger(__name__)

import os as _os

MAX_TURNS            = int(_os.getenv("MAX_TURNS",     "30"))
MAX_OBS_CHARS        = int(_os.getenv("MAX_OBS_CHARS", "2500"))
CONTEXT_KEEP         = int(_os.getenv("CONTEXT_KEEP",  "8"))
TASK_TIMEOUT_SECONDS = 280  # Hard cutoff buffer — leaves 20s for graceful exit


class AgentLoop:

    def __init__(self, llm_client: LLMClient, docker_bridge: DockerBridge, test_engine: TestEngine):
        self.llm    = llm_client
        self.docker = docker_bridge
        self.tester = test_engine

        self.system_prompt = textwrap.dedent("""\
            You are Purple Agent, an expert software engineer in a stateful Bash REPL.
            You have a STRICT time limit. Work efficiently.
            A top-tier engineer solves tasks in 5-8 carefully planned shell calls.

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
            REPO_ROOT is the value shown in the ENVIRONMENT block at the top.
            The framework appends your bash output to NOTES.txt automatically.
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
            </tools>

            <verification_rules>
            Before submitting:
            1. Run targeted tests to confirm your fix passes
            2. Handle None/null, empty lists/dicts, boundary values
            A partial fix is better than no fix.
            </verification_rules>
        """)

    def _bootstrap_workspace(self, repo_dir: str):
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
        self.docker.execute_command(f"cat > {repo_dir}/edit_file.py << 'PYEOF'\n{editor}PYEOF")

        ast_search = (
            "import sys, subprocess\n"
            "if len(sys.argv) < 2:\n"
            "    print('Usage: python ast_search.py <Name>')\n"
            "    sys.exit(1)\n"
            "target = sys.argv[1]\n"
            "exts = ['*.py','*.go','*.js','*.ts','*.tsx','*.rb','*.java','*.rs','*.c','*.cpp']\n"
            "includes = sum([['--include', e] for e in exts], [])\n"
            "pattern = rf'(def |func |class |function |fn |\\btype ).*\\b{target}\\b'\n"
            "r = subprocess.run(['grep','-rn','-E',pattern,'.']+includes, capture_output=True, text=True, cwd='.')\n"
            "if r.stdout:\n"
            "    lines = r.stdout.strip().split('\\n')\n"
            "    print(f'Found {len(lines)} definition(s):\\n\\n' + '\\n'.join(lines[:40]))\n"
            "else:\n"
            "    print('Not found.')\n"
        )
        self.docker.execute_command(f"cat > {repo_dir}/ast_search.py << 'PYEOF'\n{ast_search}PYEOF")

        test_cmd = self.tester.test_command or "echo 'No test runner found.'"
        self.docker.execute_command(
            f"printf '#!/bin/bash\\nset -e\\ncd {repo_dir}\\n{test_cmd}\\n' "
            f"> {repo_dir}/run_script.sh && chmod +x {repo_dir}/run_script.sh"
        )

        self.docker.execute_command(
            f"printf '### PURPLE AGENT NOTES ###\\n- Repo root: {repo_dir}\\n- Start of exploration.\\n' > {repo_dir}/NOTES.txt"
        )

    @staticmethod
    def _cap_observation(output: str) -> str:
        if len(output) <= MAX_OBS_CHARS:
            return output
        half = MAX_OBS_CHARS // 2
        return output[:half] + f"\n... [TRUNCATED {len(output) - MAX_OBS_CHARS} chars] ...\n" + output[-half:]

    def _append_to_notes(self, repo_dir: str, turn: int, command: str, output: str):
        summary = output.strip()[:200].replace("'", " ")
        note = f"Turn {turn}: $ {command[:80]} -> {summary}"
        self.docker.execute_command(
            f"printf '\\n{note}\\n' >> {repo_dir}/NOTES.txt 2>/dev/null || true",
            timeout=5,
        )

    def _assistant_msg(self, content: str) -> Dict:
        """
        Preserves reasoning for MiniMax continuity across turns.
        Tries reasoning_details (list, official format) first,
        falls back to reasoning (string) if that's what was returned.
        """
        msg: Dict = {"role": "assistant", "content": content}

        rd = getattr(self.llm, "_last_reasoning_details", None)
        rt = getattr(self.llm, "_last_reasoning", None)

        if rd:
            msg["reasoning_details"] = rd
        elif rt:
            msg["reasoning"] = rt  # fallback for string-format reasoning

        return msg

    @staticmethod
    def _prune_context(messages: List[Dict]) -> List[Dict]:
        if len(messages) <= 2 + CONTEXT_KEEP:
            return messages
        logger.info(
            "Context pruned: %d -> %d messages (kept system+task+last %d)",
            len(messages), 2 + CONTEXT_KEEP, CONTEXT_KEEP,
        )
        pruned = messages[:2] + messages[-CONTEXT_KEEP:]

        # MiniMax requires at least one assistant message with reasoning_details
        # to survive pruning so it can continue its reasoning chain next turn.
        # If pruning removed all of them, rescue the most recent one.
        has_reasoning = any(
            m.get("role") == "assistant" and
            (m.get("reasoning_details") or m.get("reasoning"))
            for m in pruned
        )
        if not has_reasoning:
            for m in reversed(messages):
                if m.get("role") == "assistant" and (
                    m.get("reasoning_details") or m.get("reasoning")
                ):
                    pruned.insert(-1, m)
                    logger.debug("Rescued reasoning from pruned context")
                    break

        return pruned

    async def run_stage_4_bash_repl(
        self,
        issue_text: str,
        context_primer: str,
        verify_cmd: str = "",
    ) -> Tuple[bool, List[Dict]]:

        # Global stopwatch — enforces TASK_TIMEOUT_SECONDS inside the loop
        self.start_time = time.time()

        repo_dir = self.docker.repo_dir
        self._bootstrap_workspace(repo_dir)

        messages: List[Dict] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"### ENVIRONMENT\nREPO_ROOT = `{repo_dir}`\n"
                    f"### TARGET ISSUE\n{issue_text}\n\n{context_primer}\n\n"
                    f"Begin: cd {repo_dir} && <your first diagnostic command>"
                ),
            },
        ]

        tests_run = False
        if verify_cmd and verify_cmd.strip():
            logger.info("Pre-loop: running verify_cmd: %s", verify_cmd[:80])
            ec, out = self.docker.execute_command(
                f"cd {repo_dir} && {verify_cmd}", timeout=25
            )
            out_capped = self._cap_observation(out)
            messages.append({
                "role": "user",
                "content": f"<observation status='PRE_LOOP'>Pre-flight test output:\n{out_capped}</observation>",
            })
            tests_run = ec == 0

        for turn in range(1, MAX_TURNS + 1):

            # Per-turn deadline check — force-submit before gateway kills us
            elapsed_time = time.time() - self.start_time
            if elapsed_time > TASK_TIMEOUT_SECONDS:
                logger.error(
                    "TIME LIMIT REACHED (%.1fs > %ds). Force-submitting.",
                    elapsed_time, TASK_TIMEOUT_SECONDS,
                )
                self.docker.execute_command(
                    f"cd {repo_dir} && git diff HEAD > /tmp/purple_patch.diff",
                    timeout=5,
                )
                return True, messages

            logger.info("--- TURN %d/%d (Elapsed: %.1fs) ---", turn, MAX_TURNS, elapsed_time)

            if turn == MAX_TURNS:
                self.docker.execute_command(
                    f"cd {repo_dir} && git diff HEAD > /tmp/purple_patch.diff",
                    timeout=10,
                )
                return True, messages

            messages = self._prune_context(messages)

            try:
                raw = await self.llm.generate_step(messages)
            except Exception as e:
                logger.error("LLM error turn %d: %s", turn, str(e)[:150])
                messages.append({
                    "role": "user",
                    "content": f"<observation status='FAILED'>LLM error: {str(e)[:100]}</observation>",
                })
                continue

            messages.append(self._assistant_msg(raw))

            # Write reasoning to NOTES.txt using heredoc — safe against $variables
            # in the LLM's thought output being evaluated by bash.
            reasoning = getattr(self.llm, "_last_reasoning", "")
            if reasoning and len(reasoning) > 20:
                summary = reasoning[:300]
                self.docker.execute_command(
                    f"cat << 'REASONEOF' >> {repo_dir}/NOTES.txt\n"
                    f"[Turn {turn} thinking]: {summary}...\n"
                    f"REASONEOF\n",
                    timeout=5,
                )

            try:
                thought, action_type, action_content = self.llm.parse_response(raw)
            except Exception:
                thought, action_type, action_content = "", "bash", "echo 'Parse error.'"

            logger.info("Turn %d | action=%s | content=%s", turn, action_type, action_content[:80])

            if action_type == "submit":
                if turn < 4 and not tests_run:
                    messages.append({
                        "role": "user",
                        "content": "<observation status='REJECTED'>Run tests first before submitting.</observation>",
                    })
                    continue
                return True, messages

            if action_type == "bash":
                ec, out = self.docker.execute_command(action_content)
                output_capped = self._cap_observation(out)
                self._append_to_notes(repo_dir, turn, action_content, out)

                is_test_cmd = any(kw in action_content for kw in
                                  ["pytest", "npm test", "go test", "run_script.sh"])
                if is_test_cmd:
                    tests_run = True

                is_edit = any(kw in action_content for kw in
                              ["edit_file.py", "git apply", "patch "])
                if is_edit and self.tester.test_command:
                    _, test_out = self.docker.execute_command(
                        f"cd {repo_dir} && timeout 15s {self.tester.test_command} 2>&1 | tail -20",
                        timeout=20,
                    )
                    tests_run = True
                    messages.append({
                        "role": "user",
                        "content": self.llm.format_observation(output_capped, ec),
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            f"<observation status='AUTO_TEST'>Auto-test post-edit:\n"
                            f"{self._cap_observation(test_out)}</observation>"
                        ),
                    })
                    continue

                messages.append({
                    "role": "user",
                    "content": self.llm.format_observation(output_capped, ec),
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"<observation status='FAILED'>Unknown action '{action_type}'.</observation>",
                })

        return True, messages

    async def run_stage_6_qa_phase(
        self, messages: List[Dict], max_qa_retries: int = 2
    ) -> bool:
        for attempt in range(1, max_qa_retries + 1):
            logger.info("QA attempt %d/%d", attempt, max_qa_retries)
            gate_passed, gate_msg = self.tester.verify_patch()
            if gate_passed:
                return True

            failing_tests = [
                l.strip() for l in gate_msg.splitlines()
                if l.strip().startswith("FAILED ") or l.strip().startswith("--- FAIL:")
            ]
            test_list    = "\n".join(failing_tests[:5]) if failing_tests else gate_msg[:500]
            targeted_cmd = self.tester.test_command or "git diff HEAD"

            messages.append({
                "role": "user",
                "content": (
                    f"GATE FAILED (attempt {attempt}/{max_qa_retries}).\n"
                    f"Failing:\n{test_list}\n\nRun: `{targeted_cmd}`\nFix and submit."
                ),
            })

            for _ in range(3):
                try:
                    raw = await self.llm.generate_step(self._prune_context(messages))
                except Exception as e:
                    logger.error("QA LLM error: %s", e)
                    break
                messages.append(self._assistant_msg(raw))
                try:
                    _, atype, acontent = self.llm.parse_response(raw)
                except Exception:
                    atype, acontent = "bash", "echo 'parse error'"
                if atype == "submit":
                    break
                if atype == "bash":
                    ec, out = self.docker.execute_command(acontent)
                    messages.append({
                        "role": "user",
                        "content": self.llm.format_observation(self._cap_observation(out), ec),
                    })

        return False