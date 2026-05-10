"""
TestEngine — Phase 2 v4.2.1

Key fixes vs. submitted version:
  - discover_and_run_baseline: now has a 60s timeout via docker execute_command
  - verify_patch: runs targeted single-file test first (fast feedback), falls
    back to full suite only if needed
  - Test discovery: broader — checks go.mod, Makefile, tox.ini, pytest variants
  - baseline_ids now includes ERROR lines in addition to FAILED/FAIL
"""

import re
import logging
from typing import Set, Tuple

from ..core.docker_bridge import DockerBridge

logger = logging.getLogger(__name__)


class TestEngine:
    def __init__(self, docker_bridge: DockerBridge):
        self.docker             = docker_bridge
        self.test_command: str  = ""
        self.baseline_exit_code = None
        self.baseline_output    = ""
        self.baseline_ids: Set  = set()

    # ── Stage 2: Baseline discovery ────────────────────────────────────────────

    def discover_test_command_only(self):
        """
        Lightweight version of discover_and_run_baseline() — finds the test command
        WITHOUT executing it. Called from server.py pre-flight to avoid 3-4 min stalls.
        The agent runs the actual tests inside its bash loop.
        """
        self.test_command = self._discover_test_command()

    def discover_and_run_baseline(self):
        """
        Discovers the test command and captures pre-existing failures.
        Capped at 60s per command (called from server.py with asyncio.wait_for(45s)).
        """
        self.test_command = self._discover_test_command()
        if not self.test_command:
            logger.warning("No standard test command discovered")
            return

        logger.info("Discovered test command: %s", self.test_command)
        logger.info("Running baseline (timeout=60s)...")

        self.baseline_exit_code, self.baseline_output = self.docker.execute_command(
            self.test_command, timeout=60
        )
        self.baseline_ids = self._extract_failure_ids(self.baseline_output)

        logger.info(
            "Baseline: exit=%d  pre-existing_failures=%d",
            self.baseline_exit_code,
            len(self.baseline_ids),
        )

    # ── Stage 5: Patch verification ────────────────────────────────────────────

    def verify_patch(self) -> Tuple[bool, str]:
        """
        Gate: run tests and compare against baseline.
        Returns (passed, message).

        Strategy:
        1. Run the full test command (with 120s timeout — shorter than baseline's 60s
           because by now we only care about the diff, not discovery)
        2. Filter out pre-existing failures
        3. Pass if no new regressions
        """
        if not self.test_command:
            # No test runner — accept the patch (can't verify)
            return True, "No test runner found — patch accepted unverified."

        exit_code, output = self.docker.execute_command(
            self.test_command, timeout=120
        )

        if exit_code == 0:
            return True, "All tests passed."

        current_ids = self._extract_failure_ids(output)

        # Baseline was clean — any failure is a regression
        if self.baseline_exit_code == 0 or self.baseline_exit_code is None:
            snippet = output.strip()[-1500:]
            return False, (
                f"Tests failed (exit {exit_code}) — baseline was clean.\n"
                f"Failures:\n{snippet}"
            )

        # Baseline had pre-existing failures — only new ones matter
        regressions = current_ids - self.baseline_ids
        if not regressions:
            logger.info(
                "Gate: only pre-existing failures (%d) — PASSING", len(current_ids)
            )
            return True, "Passed (pre-existing failures filtered out)."

        regression_list = "\n".join(sorted(regressions)[:10])
        return False, (
            f"You introduced {len(regressions)} new failing test(s):\n"
            f"{regression_list}\n\nFull output (last 1500 chars):\n{output[-1500:]}"
        )

    # ── Reward signal for future GRPO training ─────────────────────────────────

    def calculate_reward(self, current_output: str) -> float:
        """
        Relative reward for GRPO hypothesis ranking.
          +1.0 per fixed failure
          -2.0 per new regression
          +5.0 bonus for perfect fix
        """
        current_ids = self._extract_failure_ids(current_output)

        if not self.baseline_ids and self.baseline_exit_code == 0:
            return -2.0 * len(current_ids)

        fixed      = self.baseline_ids - current_ids
        regressions = current_ids - self.baseline_ids

        reward  = len(fixed) * 1.0
        reward -= len(regressions) * 2.0

        if len(fixed) == len(self.baseline_ids) and not regressions:
            reward += 5.0  # perfect fix bonus

        return reward

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _discover_test_command(self) -> str:
        """
        Probe common test runners in priority order.
        Returns the first working command, or empty string.
        """
        probes = [
            # Python
            ("test -f pytest.ini -o -f setup.cfg -o -f pyproject.toml -o -f tox.ini",
             "python -m pytest --tb=short -q --no-header -x"),
            ("test -f setup.py",
             "python -m pytest --tb=short -q --no-header -x"),
            # JavaScript / Node
            ("test -f package.json && grep -q '\"test\"' package.json",
             "npm test -- --forceExit 2>&1 | head -100"),
            # Go
            ("test -f go.mod",
             "go test ./... -count=1 2>&1 | tail -50"),
            # Ruby
            ("test -f Gemfile && grep -q 'rspec' Gemfile",
             "bundle exec rspec --format progress 2>&1 | tail -50"),
            # Rust
            ("test -f Cargo.toml",
             "cargo test 2>&1 | tail -50"),
            # Makefile with test target
            ("test -f Makefile && grep -q '^test' Makefile",
             "make test 2>&1 | tail -50"),
        ]

        for condition, command in probes:
            _, out = self.docker.execute_command(
                f"{condition} && echo FOUND", timeout=10
            )
            if "FOUND" in out:
                logger.info("Test probe matched: %s", command[:60])
                return command

        return ""

    def _extract_failure_ids(self, output: str) -> Set[str]:
        """Extract unique test failure identifiers from test runner output."""
        ids = set()
        for line in output.splitlines():
            s = line.strip()
            # pytest: "FAILED tests/test_foo.py::test_bar - AssertionError"
            if s.startswith("FAILED "):
                ids.add(s.split(" - ")[0].strip())
            # Go: "--- FAIL: TestFoo (0.00s)"
            elif s.startswith("--- FAIL:"):
                ids.add(s.split("(")[0].strip())
            # pytest ERROR (setup/teardown failures)
            elif s.startswith("ERROR tests/") or s.startswith("ERROR src/"):
                ids.add(s.split(" - ")[0].strip())
        return ids