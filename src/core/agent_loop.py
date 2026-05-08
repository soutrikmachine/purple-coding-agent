import logging
import asyncio
from typing import Dict, List, Any

from .llm_client import LLMClient
from .docker_bridge import DockerBridge
from ..tools.test_engine import TestEngine

logger = logging.getLogger(__name__)

class AgentLoop:
    """
    The core stateful execution engine for the Purple Agent.
    Manages the 50-turn Bash REPL (Stage 4) and the QA Fix Phase (Stage 6).
    """
    def __init__(self, llm_client: LLMClient, docker_bridge: DockerBridge, test_engine: TestEngine):
        self.llm = llm_client
        self.docker = docker_bridge
        self.test_engine = test_engine
        self.max_turns = 50
        
        self.system_prompt = (
            "You are an autonomous software engineering agent running in a stateful bash environment.\n"
            "Your task is to resolve the provided GitHub issue.\n"
            "You have full root access to a Docker container with the repository mounted at /workspace.\n"
            "You must use the following XML tags for every turn:\n"
            "<thought>Explain your reasoning here.</thought>\n"
            "<action type=\"bash\">Your bash command here</action>\n\n"
            "Special Actions:\n"
            "- <action type=\"submit\">Done</action> : Use this when you have written the fix and are ready to run the test suite."
        )

    async def run_stage_4_bash_repl(self, issue_text: str, context_primer: str) -> bool:
        """
        Executes the multi-turn REPL loop.
        Returns True if the agent explicitly submitted a solution, False if it timed out.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Issue:\n{issue_text}\n\nContext Primer:\n{context_primer}\n\nBegin."}
        ]

        turn = 0
        while turn < self.max_turns:
            turn += 1
            logger.info(f"--- Starting Turn {turn}/{self.max_turns} ---")
            
            # 1. Generate Step (using the OpenRouter Flash model configured in LLMClient)
            raw_response = await self.llm.generate_step(messages)
            
            # Append assistant's raw output to history to maintain state
            messages.append({"role": "assistant", "content": raw_response})
            
            # 2. Parse Action
            thought, action_type, action_content = self.llm.parse_response(raw_response)
            
            if not action_type:
                logger.warning("No action detected. Prompting agent to correct format.")
                messages.append({
                    "role": "user", 
                    "content": "<observation status=\"FAILED\">Error: No valid <action> tag found. Please ensure you output <action type=\"bash\">...</action>.</observation>"
                })
                continue

            # 3. Handle 'Submit'
            if action_type == "submit":
                logger.info("Agent submitted the patch. Exiting Stage 4 REPL.")
                return True
                
            # 4. Execute Bash
            if action_type == "bash":
                logger.info(f"Executing: {action_content}")
                exit_code, output = self.docker.execute_command(action_content)
                
                # 5. Format and Inject Observation
                observation = self.llm.format_observation(output, exit_code)
                messages.append({"role": "user", "content": observation})
                
            else:
                messages.append({
                    "role": "user", 
                    "content": f"<observation status=\"FAILED\">Error: Unknown action type '{action_type}'. Only 'bash' and 'submit' are supported.</observation>"
                })

        logger.warning("Max turns reached without submission.")
        return False

    async def run_stage_6_qa_phase(self, messages: List[Dict[str, str]], max_qa_retries: int = 3) -> bool:
        """
        The Mechanical Test Gate. Runs the test suite against the new patch.
        If it fails, re-injects the failure logs into the prompt for an immediate fix cycle.
        """
        retries = 0
        while retries < max_qa_retries:
            logger.info(f"--- QA Phase: Attempt {retries + 1}/{max_qa_retries} ---")
            
            # Trigger the test engine (Stage 5 smarter gate)
            test_passed, test_logs = self.test_engine.run_test_gate()
            
            if test_passed:
                logger.info("QA Phase Passed. Patch is successful.")
                return True
                
            logger.warning("QA Phase Failed. Injecting test logs back to agent.")
            qa_prompt = (
                "The test suite failed after your patch. Analyze the following logs and fix the implementation.\n\n"
                f"<test_failures>\n{test_logs}\n</test_failures>\n\n"
                "Provide your next <thought> and <action type=\"bash\"> to investigate or fix the issue."
            )
            messages.append({"role": "user", "content": qa_prompt})
            
            # Give the agent a micro-loop (e.g., 5 turns) to fix the specific QA failure
            sub_turn = 0
            while sub_turn < 5:
                sub_turn += 1
                raw_response = await self.llm.generate_step(messages)
                messages.append({"role": "assistant", "content": raw_response})
                
                thought, action_type, action_content = self.llm.parse_response(raw_response)
                
                if action_type == "submit":
                    break # Break the micro-loop to re-evaluate the test suite
                    
                if action_type == "bash":
                    exit_code, output = self.docker.execute_command(action_content)
                    observation = self.llm.format_observation(output, exit_code)
                    messages.append({"role": "user", "content": observation})
                    
            retries += 1

        logger.error("Failed to pass QA phase after maximum retries.")
        return False

    async def execute_task(self, issue_text: str, context_primer: str):
        """Main entry point to run the full task lifecycle."""
        logger.info("Initializing Agent Task.")
        
        # Stage 4
        submitted = await self.run_stage_4_bash_repl(issue_text, context_primer)
        
        # Stage 6
        if submitted:
            # Reconstruct the messages list from the LLM client's history or pass it dynamically
            # For simplicity, assuming run_stage_4_bash_repl modified a shared state, 
            # or we return the messages list from it. (Implementation detail adjusted here).
            # In a full run, you would pass the current context window into the QA phase.
            pass