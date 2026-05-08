import logging
from typing import Dict, List
from .llm_client import LLMClient
from .docker_bridge import DockerBridge
from ..tools.test_engine import TestEngine

logger = logging.getLogger(__name__)

class AgentLoop:
    """Orchestrates the stateful execution engine and QA micro-loops."""
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
            "- <action type=\"submit\">Done</action> : Use this when you have written the fix and are ready to test."
        )

    async def run_stage_4_bash_repl(self, issue_text: str, context_primer: str) -> Tuple[bool, List[Dict]]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Issue:\n{issue_text}\n\nContext Primer:\n{context_primer}\n\nBegin."}
        ]

        for turn in range(1, self.max_turns + 1):
            logger.info(f"--- Starting Turn {turn}/{self.max_turns} ---")
            
            raw_response = await self.llm.generate_step(messages)
            messages.append({"role": "assistant", "content": raw_response})
            
            thought, action_type, action_content = self.llm.parse_response(raw_response)
            
            if not action_type:
                logger.warning("No action detected. Prompting agent to correct format.")
                messages.append({
                    "role": "user", 
                    "content": "<observation status=\"FAILED\">Error: No valid <action> tag found. Please ensure you output <action type=\"bash\">...</action>.</observation>"
                })
                continue

            if action_type == "submit":
                logger.info("Agent submitted the patch. Exiting Stage 4 REPL.")
                return True, messages
                
            if action_type == "bash":
                logger.info(f"Executing: {action_content}")
                exit_code, output = self.docker.execute_command(action_content)
                observation = self.llm.format_observation(output, exit_code)
                messages.append({"role": "user", "content": observation})
            else:
                messages.append({
                    "role": "user", 
                    "content": f"<observation status=\"FAILED\">Error: Unknown action type '{action_type}'. Only 'bash' and 'submit' are supported.</observation>"
                })

        logger.warning("Max turns reached without submission.")
        return False, messages

    async def run_stage_6_qa_phase(self, messages: List[Dict[str, str]], max_qa_retries: int = 3) -> bool:
        retries = 0
        while retries < max_qa_retries:
            logger.info(f"--- QA Phase: Attempt {retries + 1}/{max_qa_retries} ---")
            
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
            
            # 5-turn Micro-loop for fixing the specific test failure
            for _ in range(5):
                raw_response = await self.llm.generate_step(messages)
                messages.append({"role": "assistant", "content": raw_response})
                
                thought, action_type, action_content = self.llm.parse_response(raw_response)
                
                if action_type == "submit":
                    break
                    
                if action_type == "bash":
                    exit_code, output = self.docker.execute_command(action_content)
                    messages.append({"role": "user", "content": self.llm.format_observation(output, exit_code)})
                    
            retries += 1

        logger.error("Failed to pass QA phase after maximum retries.")
        return False