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
    def __init__(self, docker_bridge: DockerBridge):
        self.docker = docker_bridge
        self.baseline_failures: Set[str] = set()
        self.test_framework = "pytest" # Default fallback

    def _parse_failures(self, output: str) -> Set[str]:
        """Extracts unique test IDs from logs to compute set differences."""
        failures = set()
        
        # Pytest pattern: FAIL tests/test_file.py::test_func
        pytest_pattern = r'(?:FAIL|ERROR)\s+(tests/.*|.*?\.py)::(\w+)'
        for match in re.findall(pytest_pattern, output):
            failures.add(f"{match[0]}::{match[1]}")
        
        # Unittest/Django pattern: FAIL: test_func (tests.test_file.TestCase)
        unit_pattern = r'(?:FAIL|ERROR):\s+(\w+)\s+\((.*?)\)'
        for match in re.findall(unit_pattern, output):
            failures.add(f"{match[1]}.{match[0]}")
            
        return failures

    def discover_and_run_baseline(self) -> List[str]:
        """Stage 2: Runs tests BEFORE the agent makes changes."""
        logger.info("Discovering test framework and running baseline...")
        
        # Framework detection
        exit_code, ls_out = self.docker.execute_command("ls -a")
        if "manage.py" in ls_out:
            self.test_framework = "django"
            cmd = "python manage.py test --noinput"
        elif "tox.ini" in ls_out:
            self.test_framework = "tox"
            cmd = "tox -e py39" # Common default, can be dynamically parsed
        else:
            self.test_framework = "pytest"
            cmd = "pytest -x --tb=short"

        logger.info(f"Detected framework: {self.test_framework}. Running: {cmd}")
        exit_code, output = self.docker.execute_command(cmd, timeout=300)
        
        self.baseline_failures = self._parse_failures(output)
        logger.info(f"Baseline established. Found {len(self.baseline_failures)} pre-existing failing tests.")
        
        return list(self.baseline_failures)

    def run_test_gate(self) -> Tuple[bool, str]:
        """Stage 5: Evaluates the patch by comparing current failures to baseline."""
        if self.test_framework == "django":
            cmd = "python manage.py test --noinput"
        elif self.test_framework == "tox":
            cmd = "tox"
        else:
            cmd = "pytest --tb=short"
            
        exit_code, output = self.docker.execute_command(cmd, timeout=300)
        
        if exit_code == 0:
            return True, "All tests passed successfully."
            
        current_failures = self._parse_failures(output)
        
        # Isolate new regressions caused by the agent's patch
        new_failures = current_failures - self.baseline_failures
        fixed_failures = self.baseline_failures - current_failures
        
        if not new_failures and len(fixed_failures) > 0:
            return True, f"Agent fixed tests: {list(fixed_failures)}. No new regressions."
            
        if not new_failures and len(current_failures) <= len(self.baseline_failures):
            return True, "Failures detected, but they match the baseline. Ignoring."
            
        failure_report = "\n".join(list(new_failures))
        trunc_output = output[-2000:] if len(output) > 2000 else output
        
        return False, f"NEW regressions detected:\n{failure_report}\n\nLogs:\n{trunc_output}"