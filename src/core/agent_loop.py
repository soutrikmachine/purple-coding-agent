import logging
import asyncio
import textwrap
from typing import Dict, List, Tuple, Optional
from .llm_client import LLMClient
from .docker_bridge import DockerBridge
from ..tools.test_engine import TestEngine

logger = logging.getLogger(__name__)

class AgentLoop:
    """
    The core stateful execution engine for the Purple Agent.
    Orchestrates Stage 4 (50-turn Bash REPL) and Stage 6 (QA Fix Phase).
    
    The ICLSpecialist data is introduced here as the 'context_primer' to anchor 
    the model's behavior across long-running asynchronous turns.
    """
    def __init__(self, llm_client: LLMClient, docker_bridge: DockerBridge, test_engine: TestEngine):
        self.llm = llm_client
        self.docker = docker_bridge
        self.test_engine = test_engine
        self.max_turns = 50
        
        # Base system instructions that never change across turns
        self.system_prompt = textwrap.dedent("""\
            You are Purple Agent, an expert software engineer operating autonomously.
            You are running inside a stateful Bash REPL in a Docker container with the target repository mounted at /workspace.
            
            Your objective is to solve the provided GitHub issue correctly and efficiently within a strict limit of 50 shell calls.
            Because compute resources are constrained, you must solve the problem using as few calls as possible. A top-tier solution requires 8-12 calls.

            <protocol>
            You MUST respond using this exact XML structure for every turn:
            <thought>
            Step-by-step reasoning. Analyze the problem, plan your batch reads/edits, and verify your logic against edge cases.
            </thought>
            <action type="bash">
            Your single-line bash command or chained commands here.
            </action>
            
            When you have fully verified your fix passes the tests, terminate the loop with:
            <action type="submit">Done</action>
            </protocol>

            <efficiency_and_editing>
            Minimize calls by batching your work. 
            Do NOT rely on brittle `sed` commands for multi-line edits. Instead, use Python heredocs to read, replace, and write reliably.

            Batched read example (1 call, multiple files):
              cat -n src/user/email.py | head -n 80 && echo '===FILE2===' && grep -rn 'def send' src/

            Batched edit + verify example (1 call, robust replacement):
              python -c "
            import pathlib
            f = pathlib.Path('src/api/users.py')
            content = f.read_text()
            new_content = content.replace('if not user:', 'if not user or not user.is_active:')
            f.write_text(new_content)
            " && grep -n 'is_active' src/api/users.py

            RULES FOR EDITING:
            1. The old string in `.replace()` must match EXACTLY, or it silently fails.
            2. ALWAYS chain a `grep` or `diff` immediately after your edit to verify it landed.
            3. Make MINIMAL changes. Change only the lines needed. Do not rewrite whole functions.
            </efficiency_and_editing>

            <rigorous_grading>
            After you submit, your patch will face a strict Mechanical Test Gate. 
            Code must be safety-critical. Every modified or new function MUST handle:
            - null, undefined, or None values
            - empty arrays/lists/dicts
            - missing object keys
            - boundary conditions (0, -1, max limits)
            
            Prioritize core requirements and robust edge-case handling. Ignore cosmetic improvements or peripheral linting.
            </rigorous_grading>

            <self_test_before_submit>
            Before you issue <action type="submit">, you must look past the obvious symptom:
            1. Run the local test suite (e.g., `pytest tests/path_to_test.py -x --tb=short`).
            2. If tests are too slow, write a quick sanity check (`python -c "import module; module.test_func()"`) to verify your fix.
            3. Does your fix handle the EMPTY case? The NULL case?
            4. Review neighboring code. Your fix must match the surrounding error-handling patterns.
            
            Finding and fixing failures yourself using bash is cheaper than having your patch rejected by the final QA gate.
            </self_test_before_submit>
        """)

    async def run_stage_4_bash_repl(self, issue_text: str, context_primer: str) -> Tuple[bool, List[Dict]]:
        """
        Executes the 50-turn stateful loop.
        
        The 'context_primer' contains the Stage 2.5 ICL Injection:
        - Domain-specific rules (Django, Pytest, etc.)
        - Few-shot examples of correct thought/action/observation loops
        - Diagnostic hypotheses
        """
        # Initializing history with the ICL-enriched primer
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                f"### TARGET ISSUE\n{issue_text}\n\n"
                f"{context_primer}\n\n"
                "Please begin your exploration by verifying the issue."
            )}
        ]

        logger.info(f"Starting Stage 4 Bash REPL with {self.max_turns} turn limit.")

        for turn in range(1, self.max_turns + 1):
            logger.info(f"--- TURN {turn}/{self.max_turns} ---")
            
            try:
                # 1. Generate turn using DeepSeek-v4-flash via OpenRouter
                raw_response = await self.llm.generate_step(messages)
                messages.append({"role": "assistant", "content": raw_response})
                
                # 2. Parse the LLM's intent
                thought, action_type, action_content = self.llm.parse_response(raw_response)

                try:
                    thought, action_type, action_content = self.llm.parse_response(raw_response)
                except Exception as e:
                    logger.error(f"Catastrophic parsing failure on turn {turn}: {e}")
                    thought = "System recovered from parsing crash."
                    action_type = "bash"
                    action_content = f"echo 'System Error: Parser exception {str(e)}. You MUST use strict XML format: <action type=\"bash\">your command</action>'"
                
                if not action_type:
                    logger.warning("Agent failed to provide an action. Requesting retry.")
                    messages.append({
                        "role": "user", 
                        "content": "<observation status=\"FAILED\">Error: Missing <action> tag. Please provide a command.</observation>"
                    })
                    continue

                # 3. Handle 'Submit' - transition to Stage 6
                if action_type == "submit":
                    logger.info("Agent issued 'submit'. Terminating REPL loop.")
                    return True, messages
                    
                # 4. Handle 'Bash' execution inside the Sibling Container
                if action_type == "bash":
                    logger.info(f"Action [Bash]: {action_content}")
                    exit_code, output = self.docker.execute_command(action_content)
                    
                    # 5. Inject ground-truth observation back into context
                    observation = self.llm.format_observation(output, exit_code)
                    messages.append({"role": "user", "content": observation})
                else:
                    messages.append({
                        "role": "user", 
                        "content": f"<observation status=\"FAILED\">Error: Unknown action '{action_type}'. Use 'bash' or 'submit'.</observation>"
                    })
                    
            except Exception as e:
                logger.error(f"Critical execution error in REPL turn {turn}: {e}")
                # We do not break here; allow the loop to continue and potentially recover
                messages.append({
                    "role": "user",
                    "content": f"<observation status=\"FAILED\">Critical internal error: {str(e)}.</observation>"
                })

        logger.warning("Agent reached maximum turns without submitting.")
        return False, messages

    async def run_stage_6_qa_phase(self, messages: list, max_qa_retries: int = 3) -> bool:
        """
        Stage 6: The Mechanical Test Gate micro-loop with TARGETED FEEDBACK.
        This provides immediate feedback and forces the LLM to run isolated, rapid tests.
        """
        for attempt in range(1, max_qa_retries + 1):
            logger.info(f"--- QA GATE ATTEMPT {attempt}/{max_qa_retries} ---")
            
            # 1. Execute Smarter Gate (Stage 5 logic - Secret 6)
            # Note: Ensure your AgentLoop __init__ maps the test engine to self.tester 
            # (e.g., self.tester = tester) to match server.py
            gate_passed, gate_msg = self.test_engine.verify_patch()
            
            if gate_passed:
                logger.info("QA Gate Passed. Solution is viable and regression-free.")
                return True
                
            logger.warning(f"QA Gate Failed on attempt {attempt}. Injecting targeted logs for repair.")
            
            # 2. Anchor the correction using the TARGETED FEEDBACK concept
            qa_instruction = (
                f"CRITICAL: The broad test suite failed. Here are the specific tests that are failing:\n"
                f"{gate_msg}\n\n"
                f"CRITICAL INSTRUCTION: Do NOT run the entire test suite again. It is too slow. "
                f"Use your bash shell to run ONLY the specific failing tests isolated above "
                f"(e.g., `pytest path/to/test_file.py::test_specific_function -x` or `npm test -- -t 'test_name'`).\n"
                f"Iterate rapidly using these targeted tests until they pass. Only use the 'submit' action when resolved."
            )
            messages.append({"role": "user", "content": qa_instruction})
            
            # 3. Allow a 5-turn 'repair burst' per QA failure
            for sub_turn in range(5):
                raw_response = await self.llm.generate_step(messages)
                messages.append({"role": "assistant", "content": raw_response})
                
                # PROTECTED: Safe parsing logic mirrored for the inner repair loop
                try:
                    _, action_type, action_content = self.llm.parse_response(raw_response)
                except Exception as e:
                    logger.error(f"Catastrophic parsing failure in QA loop: {e}")
                    action_type = "bash"
                    action_content = f"echo 'System Error: Parser exception {str(e)}. You MUST use strict XML format: <action type=\"bash\">your command</action>'"
                
                if action_type == "submit":
                    break # Break inner loop to re-run the main broad test gate
                
                if action_type == "bash":
                    exit_code, output = self.docker.execute_command(action_content)
                    messages.append({"role": "user", "content": self.llm.format_observation(output, exit_code)})

        logger.error("QA Gate failed after maximum retries. Patch rejected.")
        return False