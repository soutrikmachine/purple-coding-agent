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
import time  # <-- NEW: Required for the 300s deadline stopwatch
from typing import Dict, List, Tuple

from .llm_client import LLMClient
from .docker_bridge import DockerBridge
from ..tools.test_engine import TestEngine

logger = logging.getLogger(__name__)

import os as _os

MAX_TURNS     = int(_os.getenv("MAX_TURNS",     "14"))
MAX_OBS_CHARS = int(_os.getenv("MAX_OBS_CHARS", "1500"))
CONTEXT_KEEP  = int(_os.getenv("CONTEXT_KEEP",  "8"))
TASK_TIMEOUT_SECONDS = 280  # <-- NEW: Hard cutoff buffer (leaves 20s for graceful exit)


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
        # (Your existing bootstrap code remains exactly the same)
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
        FIX 1: Captures OpenRouter's new standard 'reasoning' field if 'reasoning_details' 
        is missing. This ensures MiniMax does not lose its chain of thought.
        """
        msg: Dict = {"role": "assistant", "content": content}
        
        rd = getattr(self.llm, "_last_reasoning_details", None)
        rt = getattr(self.llm, "_last_reasoning", None)
        
        if rd:
            msg["reasoning_details"] = rd
        elif rt:
            msg["reasoning"] = rt  # The new standard OpenAI formatting payload
            
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
        
        # --- NEW: THE GLOBAL STOPWATCH ---
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
            # REDUCED TIMEOUT: We cannot afford 60s on pre-flight if total budget is 300s
            ec, out = self.docker.execute_command(f"cd {repo_dir} && {verify_cmd}", timeout=25) 
            out_capped = self._cap_observation(out)
            messages.append({
                "role": "user",
                "content": f"<observation status='PRE_LOOP'>Pre-flight test output:\n{out_capped}</observation>",
            })
            tests_run = ec == 0

        for turn in range(1, MAX_TURNS + 1):
            # --- NEW: TIMEOUT ENFORCER ---
            elapsed_time = time.time() - self.start_time
            if elapsed_time > TASK_TIMEOUT_SECONDS:
                logger.error(f"TIME LIMIT REACHED ({elapsed_time:.1f}s > {TASK_TIMEOUT_SECONDS}s). Force-submitting to save score!")
                self.docker.execute_command(f"cd {repo_dir} && git diff HEAD > /tmp/purple_patch.diff", timeout=5)
                return True, messages

            logger.info("--- TURN %d/%d (Elapsed: %.1fs) ---", turn, MAX_TURNS, elapsed_time)

            if turn == MAX_TURNS:
                self.docker.execute_command(f"cd {repo_dir} && git diff HEAD", timeout=10)
                return True, messages

            messages = self._prune_context(messages)

            try:
                raw = await self.llm.generate_step(messages)
            except Exception as e:
                logger.error("LLM error turn %d: %s", turn, str(e)[:150])
                messages.append({"role": "user", "content": f"<observation status='FAILED'>LLM error: {str(e)[:100]}</observation>"})
                continue

            messages.append(self._assistant_msg(raw))

            # --- FIX: SAFE REASONING WRITE ---
            # Using heredoc 'EOF' prevents bash from evaluating $variables inside the LLM's thought process.
            reasoning = getattr(self.llm, "_last_reasoning", "")
            if reasoning and len(reasoning) > 20:
                summary = reasoning[:300]
                self.docker.execute_command(
                    f"cat << 'EOF' >> {repo_dir}/NOTES.txt\n[Turn {turn} thinking]: {summary}...\nEOF\n",
                    timeout=5,
                )

            try:
                thought, action_type, action_content = self.llm.parse_response(raw)
            except Exception:
                thought, action_type, action_content = "", "bash", "echo 'Parse error.'"

            if action_type == "submit":
                if turn < 4 and not tests_run:
                    messages.append({"role": "user", "content": "<observation status='REJECTED'>Run tests first.</observation>"})
                    continue
                return True, messages

            if action_type == "bash":
                ec, out = self.docker.execute_command(action_content)
                output_capped = self._cap_observation(out)
                self._append_to_notes(repo_dir, turn, action_content, out)

                is_test_cmd = any(kw in action_content for kw in ["pytest", "npm test", "go test", "run_script.sh"])
                if is_test_cmd: tests_run = True

                is_edit = any(kw in action_content for kw in ["edit_file.py", "git apply", "patch "])
                if is_edit and self.tester.test_command:
                    # REDUCED TIMEOUT: Protect the 300s budget
                    _, test_out = self.docker.execute_command(
                        f"cd {repo_dir} && timeout 15s {self.tester.test_command} 2>&1 | tail -20",
                        timeout=20, 
                    )
                    tests_run = True
                    messages.append({"role": "user", "content": self.llm.format_observation(output_capped, ec)})
                    messages.append({
                        "role": "user",
                        "content": f"<observation status='AUTO_TEST'>Auto-test post-edit:\n{self._cap_observation(test_out)}</observation>"
                    })
                    continue

                messages.append({"role": "user", "content": self.llm.format_observation(output_capped, ec)})

        return True, messages