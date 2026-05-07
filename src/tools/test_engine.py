import logging
import re
from typing import List, Dict, Set, Tuple

logger = logging.getLogger("purple_agent.test_engine")

class TestEngine:
    """
    Handles test discovery, baseline capture (Stage 2), and the mechanical 
    test gate verification (Stage 5) for the agent loop.
    """

    def __init__(self, docker_bridge):
        self.bridge = docker_bridge
        self.baseline_failures: Set[str] = set()
        self.test_command: str = ""

    def discover_and_capture_baseline(self) -> Tuple[bool, str]:
        """
        Stage 2: Discovers how to run tests, runs them, and records the baseline 
        so the agent doesn't chase pre-existing broken tests.
        """
        logger.info("Stage 2: Discovering test command and capturing baseline...")
        
        # 1. Heuristic Discovery (We can expand this list based on domain)
        # Often SWE-bench supplies a test patch, but we need a command to trigger it.
        probes = [
            ("pytest", "python -m pytest --tb=short"),
            ("manage.py", "python manage.py test"),
            ("package.json", "npm test"),
            ("go.mod", "go test ./..."),
            ("Cargo.toml", "cargo test")
        ]

        # Check which ecosystem files exist
        for marker, cmd in probes:
            exit_code, _ = self.bridge.execute_bash(f"test -f {marker}")
            if exit_code == 0:
                self.test_command = cmd
                break
        
        # Default fallback for Python if nothing explicitly matches
        if not self.test_command:
            logger.warning("No explicit test marker found. Defaulting to pytest.")
            self.test_command = "python -m pytest --tb=short"

        # 2. Run the baseline
        logger.info(f"Running baseline tests with command: {self.test_command}")
        exit_code, output = self.bridge.execute_bash(self.test_command, timeout=180)
        
        stdout = output.get("stdout", "")
        
        # 3. Parse baseline failures
        self.baseline_failures = set(self._parse_failures(stdout))
        
        logger.info(f"Baseline captured. Found {len(self.baseline_failures)} pre-existing failures.")
        return True, stdout

    def run_smart_gate(self, specific_test: str = None) -> Tuple[bool, str, List[str]]:
        """
        Stage 5: The Mechanical Test Gate.
        Runs the tests and compares the new failures against the baseline.
        
        Args:
            specific_test: If provided, runs ONLY this test (e.g., 'pytest path/to/test.py::test_name')
                           This is the "Smarter test gate" for fast feedback.
                           
        Returns:
            (passed_gate, raw_output, new_failures_list)
        """
        cmd = specific_test if specific_test else self.test_command
        logger.info(f"Stage 5: Running Test Gate -> {cmd}")
        
        exit_code, output = self.bridge.execute_bash(cmd, timeout=180)
        stdout = output.get("stdout", "")
        
        # If exit code is 0, tests passed cleanly!
        if exit_code == 0:
            return True, stdout, []

        # Parse current failures
        current_failures = self._parse_failures(stdout)
        
        # Filter out baseline failures
        new_failures = [f for f in current_failures if f not in self.baseline_failures]
        
        # If there are failures, but they were ALL in the baseline, the patch didn't break 
        # anything new, and might have fixed the target bug (though ideally the target bug 
        # transitions from baseline-fail to pass).
        # For SWE-bench, usually we are looking for a transition of the *target* test.
        if not new_failures:
            logger.info("Test run failed, but NO NEW failures introduced beyond baseline.")
            # Depending on strictness, you might consider this a pass if the target bug is fixed.
            # For now, we return the raw output so the LLM can decide.
            return False, stdout, new_failures

        logger.warning(f"Test Gate FAILED. Introduced {len(new_failures)} new failures.")
        return False, stdout, new_failures

    def _parse_failures(self, stdout: str) -> List[str]:
        """
        Lightweight parser to extract failing test IDs. 
        Currently tuned for Pytest, which is 90% of SWE-bench.
        """
        failures = []
        
        # Look for Pytest failure lines: "FAILED path/to/test.py::test_function"
        # or "ERROR path/to/test.py"
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("FAILED ") or line.startswith("ERROR "):
                # Extract the test path/name
                parts = line.split(" ", 1)
                if len(parts) > 1:
                    # Clean up trailing info like "- AssertionError: ..."
                    test_id = parts[1].split(" - ")[0].strip()
                    failures.append(test_id)
                    
        return list(set(failures))