"""
Note: This module acts as the Mechanical Test Gate.
In Phase 2, we don't guess if a patch works using a PRM; 
we run the tests inside the DockerBridge and compare 
current failures against the baseline.
"""

import re
import logging
from typing import Set, Tuple, List
from ..core.docker_bridge import DockerBridge

logger = logging.getLogger(__name__)

class TestEngine:
    def __init__(self, docker_bridge):
        self.docker = docker_bridge
        self.test_command = None
        self.baseline_exit_code = None
        self.baseline_output = ""
        self.baseline_ids = set()

    def discover_and_run_baseline(self):
        """Stage 2: Find the test command and capture pre-existing failures."""
        self.test_command = self._discover_test_command()
        
        if not self.test_command:
            logger.warning("No standard test command discovered.")
            return

        logger.info(f"Discovered test command: {self.test_command}")
        logger.info("Running baseline tests...")
        
        self.baseline_exit_code, self.baseline_output = self.docker.execute_command(self.test_command, timeout=300)
        self.baseline_ids = self._extract_failure_ids(self.baseline_output)
        
        if self.baseline_exit_code != 0:
            logger.info(f"Baseline tests have pre-existing failures ({len(self.baseline_ids)} found). Filter applied.")
        else:
            logger.info("Baseline tests are totally clean.")

    def calculate_reward(self, current_output: str) -> float:
        """Stage 3: Calculates Relative Reward Signal for GRPO hypothesis ranking."""
        if not self.baseline_ids and self.baseline_exit_code == 0:
            current_ids = self._extract_failure_ids(current_output)
            return -2.0 * len(current_ids)

        current_ids = self._extract_failure_ids(current_output)
        
        new_failures = current_ids - self.baseline_ids
        fixed_failures = self.baseline_ids - current_ids

        reward = 0.0
        reward += len(fixed_failures) * 1.0  # +1 for every fixed test
        reward -= len(new_failures) * 2.0    # -2 for every new regression
        
        if len(fixed_failures) == len(self.baseline_ids) and len(new_failures) == 0:
            reward += 5.0 # Bonus for perfect fix
            
        return reward

    def verify_patch(self) -> tuple[bool, str]:
        """Stage 6 Gate: Runs tests and compares against baseline for Pass/Fail."""
        if not self.test_command:
            return True, "No test command found, assuming manual verification."

        exit_code, output = self.docker.execute_command(self.test_command, timeout=300)
        
        if exit_code == 0:
            return True, "Tests passed."

        if self.baseline_exit_code == 0 or self.baseline_exit_code is None:
            return False, f"Tests failed with exit code {exit_code}.\nOutput:\n{output[:2000]}"

        new_ids = self._extract_failure_ids(output)
        regressions = new_ids - self.baseline_ids

        if len(regressions) == 0 and len(new_ids) > 0:
            logger.info("Test gate failed, but only due to pre-existing baseline failures. PASSING agent.")
            return True, "Passed (pre-existing failures filtered)."
        
        regression_msg = "\n".join(list(regressions)[:10])
        return False, f"You introduced NEW failing tests:\n{regression_msg}\n\nFull output:\n{output[:2000]}"

    def _discover_test_command(self) -> str:
        if "found" in self.docker.execute_command("test -f pytest.ini -o -f setup.cfg -o -f pyproject.toml && echo found")[1]:
            return "python -m pytest --tb=short -q"
        if "found" in self.docker.execute_command("test -f package.json && grep -q '\"test\"' package.json && echo found")[1]:
            return "npm test"
        if "found" in self.docker.execute_command("test -f go.mod && echo found")[1]:
            return "go test ./..."
        return None

    def _extract_failure_ids(self, output: str) -> set:
        ids = set()
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("FAILED "):
                ids.add(stripped.split(" - ")[0].strip())
            elif stripped.startswith("--- FAIL:"):
                ids.add(stripped.split("(")[0].strip())
            elif stripped.startswith("ERROR "):
                ids.add(stripped)
        return ids