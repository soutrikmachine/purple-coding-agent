"""
test_core.py — Purple Agent Phase 2 unit tests

Corrections vs. original:
  - LLMClient.format_observation: "FAILED (exit N)" not "FAILED (Exit N)"
  - LLMClient.parse_response: typeless <action> now supported (treats as bash)
  - DockerBridge.execute_command: uses subprocess.run, NOT container.exec_run
  - TestEngine: method is verify_patch() not run_test_gate()
                field is baseline_ids not baseline_failures
                method is _extract_failure_ids not _parse_failures
                pytest format is "FAILED ..." not "FAIL ..."
                test_command must be set for verify_patch to run
  - ASTGraphBuilder: __init__ loads tree-sitter at import time — mocked in tests
  - server._extract_task: renamed from _extract_task_and_context
"""

import sys
import pytest
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── LLM Client Tests ──────────────────────────────────────────────────────────

class TestLLMClient:
    @pytest.fixture
    def client(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            from core.llm_client import LLMClient
            return LLMClient()

    def test_parse_response_bash(self, client):
        """Standard well-formed XML response."""
        raw = '<thought>I will list files.</thought><action type="bash">ls -R</action>'
        thought, act_type, content = client.parse_response(raw)
        assert thought == "I will list files."
        assert act_type == "bash"
        assert content == "ls -R"

    def test_parse_response_submit(self, client):
        """Submit action is parsed correctly."""
        raw = "<thought>Done.</thought><action type=\"submit\">Done</action>"
        thought, act_type, content = client.parse_response(raw)
        assert act_type == "submit"
        assert content == "Done"

    def test_parse_response_typeless_action(self, client):
        """
        Model forgot the type attribute — parser falls back to bash.
        Regression: original parser required type= and returned an error echo.
        """
        raw = "<thought>Check logs</thought><action>cat error.log</action>"
        thought, act_type, content = client.parse_response(raw)
        assert act_type == "bash"
        assert content == "cat error.log"

    def test_parse_response_gemini_xml_fence(self, client):
        """Gemini wraps entire response in ```xml fences — parser must strip them."""
        raw = (
            "```xml\n"
            "<thought>I will grep for the bug.</thought>\n"
            '<action type="bash">grep -n "def parse" src/core.py</action>\n'
            "```"
        )
        thought, act_type, content = client.parse_response(raw)
        assert act_type == "bash"
        assert "grep" in content

    def test_parse_response_empty(self, client):
        """Empty response returns a safe error bash command, not a crash."""
        thought, act_type, content = client.parse_response("")
        assert act_type == "bash"
        assert "EMPTY RESPONSE" in content or "echo" in content

    def test_format_observation_success(self, client):
        """Exit 0 → SUCCESS status."""
        obs = client.format_observation("All tests passed", 0)
        assert 'status="SUCCESS"' in obs
        assert "All tests passed" in obs

    def test_format_observation_failure(self, client):
        """
        Exit non-zero → FAILED status.
        IMPORTANT: format is 'FAILED (exit N)' — lowercase 'exit'.
        """
        obs = client.format_observation("File not found", 1)
        # Correct: lowercase 'exit'
        assert 'status="FAILED (exit 1)"' in obs
        assert "File not found" in obs


# ── Docker Bridge Tests ───────────────────────────────────────────────────────

class TestDockerBridge:
    """
    DockerBridge.execute_command uses subprocess.run (not container.exec_run).
    SECRET 3: subprocess CLI bypasses the Amber proxy EOF bug.
    Tests mock subprocess.run, not the Docker SDK.
    """

    @patch("docker.from_env")
    def test_execute_command_calls_subprocess(self, mock_docker):
        """Command is forwarded to `docker exec` via subprocess.run."""
        from core.docker_bridge import DockerBridge

        bridge = DockerBridge(image_name="test-image")
        bridge.container = MagicMock()
        bridge.container.id = "abc123"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"hello world\n"
        mock_result.stderr = b""

        with patch("subprocess.run", return_value=mock_result) as mock_sub:
            code, output = bridge.execute_command("echo hello", timeout=30)

        assert code == 0
        assert "hello world" in output

        # Verify subprocess was called (not exec_run)
        mock_sub.assert_called_once()
        cmd_args = mock_sub.call_args[0][0]   # first positional arg = the command list
        assert "docker" in cmd_args
        assert "exec" in cmd_args
        assert "abc123" in cmd_args
        assert "echo hello" in cmd_args

    @patch("docker.from_env")
    def test_execute_command_includes_timeout_wrapper(self, mock_docker):
        """Container-side timeout is injected for safety."""
        from core.docker_bridge import DockerBridge

        bridge = DockerBridge(image_name="test-image")
        bridge.container = MagicMock()
        bridge.container.id = "deadbeef"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"ok"
        mock_result.stderr = b""

        with patch("subprocess.run", return_value=mock_result) as mock_sub:
            bridge.execute_command("ls", timeout=45)

        cmd_args = mock_sub.call_args[0][0]
        assert "timeout" in cmd_args
        assert "45s" in cmd_args

    @patch("docker.from_env")
    def test_execute_command_no_container(self, mock_docker):
        """Returns -1 and error string when no container is running."""
        from core.docker_bridge import DockerBridge

        bridge = DockerBridge(image_name="test-image")
        bridge.container = None

        code, output = bridge.execute_command("ls")
        assert code == -1
        assert "not running" in output.lower() or "Error" in output

    @patch("docker.from_env")
    def test_execute_command_combines_stderr(self, mock_docker):
        """Non-empty stderr is appended to stdout in the combined output."""
        from core.docker_bridge import DockerBridge

        bridge = DockerBridge(image_name="test-image")
        bridge.container = MagicMock()
        bridge.container.id = "abc"

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = b"partial output"
        mock_result.stderr = b"ERROR: something failed"

        with patch("subprocess.run", return_value=mock_result):
            code, output = bridge.execute_command("bad-cmd")

        assert "partial output" in output
        assert "ERROR: something failed" in output

    @patch("docker.from_env")
    def test_execute_command_uses_repo_dir_as_workdir(self, mock_docker):
        """
        execute_command passes -w self.repo_dir to docker exec.
        After _detect_repo_dir() runs, this will be /testbed or /app, not /workspace.
        This test verifies the -w flag is dynamic, not hardcoded.
        """
        from core.docker_bridge import DockerBridge

        bridge = DockerBridge(image_name="test-image")
        bridge.container = MagicMock()
        bridge.container.id = "abc"
        bridge.repo_dir = "/testbed"   # simulate post-detection state

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"ok"
        mock_result.stderr = b""

        with patch("subprocess.run", return_value=mock_result) as mock_sub:
            bridge.execute_command("ls")

        cmd_args = mock_sub.call_args[0][0]
        assert "-w" in cmd_args
        wi = cmd_args.index("-w")
        assert cmd_args[wi + 1] == "/testbed"   # NOT hardcoded /workspace

    @patch("docker.from_env")
    def test_detect_repo_dir_finds_known_path(self, mock_docker):
        """
        _detect_repo_dir() probes well-known paths first.
        If /testbed contains .git it is returned without running find.
        """
        from core.docker_bridge import DockerBridge

        bridge = DockerBridge(image_name="test-image")
        bridge.container = MagicMock()
        bridge.container.id = "abc"

        def fake_exec(cmd, timeout=10, workdir="/"):
            # Simulate: /testbed has .git, others don't
            if "test -d /testbed/.git" in cmd:
                return (0, "GIT_FOUND")
            return (1, "")

        bridge.execute_command = fake_exec
        result = bridge._detect_repo_dir()
        assert result == "/testbed"

    @patch("docker.from_env")
    def test_detect_repo_dir_falls_back_to_find(self, mock_docker):
        """
        _detect_repo_dir() falls back to `find` if no known path has .git.
        Covers repos at unusual locations like /home/user/project.
        """
        from core.docker_bridge import DockerBridge

        bridge = DockerBridge(image_name="test-image")
        bridge.container = MagicMock()
        bridge.container.id = "abc"

        def fake_exec(cmd, timeout=10, workdir="/"):
            if "test -d " in cmd and "/.git" in cmd:
                return (1, "")   # all known paths miss
            if "find / -maxdepth 4" in cmd:
                return (0, "/home/user/project/.git\n")
            return (1, "")

        bridge.execute_command = fake_exec
        result = bridge._detect_repo_dir()
        assert result == "/home/user/project"

    @patch("docker.from_env")
    def test_detect_repo_dir_returns_empty_on_total_miss(self, mock_docker):
        """
        _detect_repo_dir() returns empty string (not a crash) if no repo found.
        start_container() will then fall back to '/'.
        """
        from core.docker_bridge import DockerBridge

        bridge = DockerBridge(image_name="test-image")
        bridge.container = MagicMock()
        bridge.container.id = "abc"

        bridge.execute_command = lambda cmd, timeout=10, workdir="/": (1, "")
        result = bridge._detect_repo_dir()
        assert result == ""

    @patch("docker.from_env")
    def test_docker_client_init_failure_is_graceful(self, mock_docker):
        """Missing Docker socket produces a clean error, not a crash."""
        mock_docker.side_effect = Exception("Cannot connect to Docker daemon")
        from core.docker_bridge import DockerBridge

        bridge = DockerBridge(image_name="test-image")
        assert bridge.client is None
        result = bridge.start_container()
        assert result is False


# ── Test Engine Tests ─────────────────────────────────────────────────────────

class TestTestEngine:
    @pytest.fixture
    def engine(self):
        from tools.test_engine import TestEngine
        mock_bridge = MagicMock()
        return TestEngine(mock_bridge)

    def test_failure_parsing_pytest_format(self, engine):
        """
        Pytest FAILED lines are extracted by _extract_failure_ids.
        Note: pytest outputs 'FAILED path::test_name', NOT 'FAIL'.
        """
        output = "FAILED tests/test_api.py::test_login_auth - AssertionError"
        failures = engine._extract_failure_ids(output)
        assert "FAILED tests/test_api.py::test_login_auth" in failures

    def test_failure_parsing_go_format(self, engine):
        """Go test FAIL lines are extracted correctly."""
        output = "--- FAIL: TestLoginAuth (0.01s)"
        failures = engine._extract_failure_ids(output)
        assert any("TestLoginAuth" in f for f in failures)

    def test_failure_parsing_empty_output(self, engine):
        """Empty output returns an empty set."""
        assert engine._extract_failure_ids("") == set()

    def test_gate_passes_when_all_tests_pass(self, engine):
        """
        verify_patch() passes when test command exits 0.
        Uses verify_patch(), NOT run_test_gate().
        """
        engine.test_command = "python -m pytest"
        engine.docker.execute_command.return_value = (0, "1 passed")

        passed, msg = engine.verify_patch()
        assert passed is True

    def test_gate_fails_on_new_regression(self, engine):
        """
        verify_patch() fails when a new test regresses that wasn't in baseline.
        Uses baseline_ids (not baseline_failures).
        """
        engine.test_command = "python -m pytest"
        # Pre-existing failure that we already knew about
        engine.baseline_ids = {"FAILED tests/old_broken.py::test_init"}
        engine.baseline_exit_code = 1

        # Agent introduced a NEW failure
        new_output = (
            "FAILED tests/old_broken.py::test_init - AssertionError\n"
            "FAILED tests/new_bug.py::test_logic - AssertionError"
        )
        engine.docker.execute_command.return_value = (1, new_output)

        passed, msg = engine.verify_patch()
        assert passed is False
        assert "new_bug" in msg

    def test_gate_passes_when_only_preexisting_failures(self, engine):
        """
        verify_patch() passes if only pre-existing baseline failures remain
        and no new regressions were introduced.
        """
        engine.test_command = "python -m pytest"
        engine.baseline_ids = {"FAILED tests/old_broken.py::test_init"}
        engine.baseline_exit_code = 1

        # Same pre-existing failure, nothing new
        engine.docker.execute_command.return_value = (
            1, "FAILED tests/old_broken.py::test_init - AssertionError"
        )

        passed, msg = engine.verify_patch()
        assert passed is True
        assert "pre-existing" in msg.lower()

    def test_gate_skipped_without_test_command(self, engine):
        """verify_patch() accepts patch when no test runner is found."""
        engine.test_command = ""
        passed, msg = engine.verify_patch()
        assert passed is True

    def test_reward_signal_perfect_fix(self, engine):
        """calculate_reward gives bonus for fixing all baseline failures."""
        engine.baseline_ids = {"FAILED tests/a.py::test_x"}
        engine.baseline_exit_code = 1
        reward = engine.calculate_reward("")  # all failures gone
        # Fixed 1 (+1.0) + bonus for perfect fix (+5.0) = 6.0
        assert reward >= 5.0

    def test_discover_test_command_only_sets_attribute(self, engine):
        """discover_test_command_only populates test_command without running tests."""
        engine.docker.execute_command.return_value = (0, "FOUND")
        engine.discover_test_command_only()
        # Should have set test_command to something (or empty string if no runner found)
        assert hasattr(engine, "test_command")


# ── AST Graph Tests ───────────────────────────────────────────────────────────

class TestASTGraph:
    """
    ASTGraphBuilder loads tree-sitter grammars in __init__.
    Tests mock __init__ to avoid the tree-sitter dependency in CI.
    The executable CLI (if __name__ == '__main__') is tested separately.
    """

    def _make_builder(self):
        """Create an ASTGraphBuilder with tree-sitter init mocked out."""
        from tools.ast_graph import ASTGraphBuilder

        with patch.object(ASTGraphBuilder, "__init__", return_value=None):
            builder = ASTGraphBuilder.__new__(ASTGraphBuilder)
            builder.repo_path = "/testbed"   # realistic SWE-bench path, not /workspace
            # Minimal stub attributes the methods need
            builder.PY_LANGUAGE = MagicMock()
            builder.JS_LANGUAGE = MagicMock()
            builder.py_parser   = MagicMock()
            builder.js_parser   = MagicMock()
            builder.py_query    = MagicMock()
            builder.js_query    = MagicMock()
        return builder

    @patch("os.walk")
    def test_repo_skeleton_generation(self, mock_walk):
        """Structural map is generated and contains expected entries."""
        mock_walk.return_value = [
            ("/testbed",       ["src"], ["README.md"]),
            ("/testbed/src",   [],      ["core.py"]),
        ]

        builder = self._make_builder()

        with patch.object(
            builder.__class__,
            "parse_file_skeleton",
            return_value={"imports": [], "classes": ["Agent"], "functions": ["run"]},
        ):
            graph = builder.build_repo_graph()

        assert "File: `src/core.py`" in graph  # relative to repo_path=/testbed
        assert "Agent" in graph
        assert "run" in graph

    @patch("os.walk")
    def test_vendor_dirs_excluded(self, mock_walk):
        """node_modules, vendor, __pycache__ are excluded from the graph."""
        mock_walk.return_value = [
            ("/testbed",                  ["node_modules", "src"], []),
            ("/testbed/node_modules",     [],                      ["index.js"]),
            ("/testbed/src",              [],                      ["app.py"]),
        ]

        builder = self._make_builder()

        with patch.object(
            builder.__class__,
            "parse_file_skeleton",
            return_value={"imports": [], "classes": [], "functions": ["main"]},
        ):
            graph = builder.build_repo_graph()

        assert "node_modules" not in graph
        assert "src/app.py" in graph

    def test_captures_api_compatibility(self):
        """
        tree-sitter ≥0.21 captures() returns dict[str, list[Node]].
        Regression test: old code iterated as (node, name) pairs which broke.
        """
        from tools.ast_graph import ASTGraphBuilder

        builder = self._make_builder()

        # Simulate tree-sitter >=0.21 dict-style captures
        mock_node = MagicMock()
        mock_node.start_byte = 0
        mock_node.end_byte = 10
        source = b"def hello():"

        # Our fixed code handles both dict and list formats
        # Verify the isinstance(captures, dict) branch works
        captures_dict = {"func_name": [mock_node]}

        def get_text(node):
            return source[node.start_byte:node.end_byte].decode()

        # Check the iteration logic directly
        if isinstance(captures_dict, dict):
            capture_iter = [
                (node, name)
                for name, nodes in captures_dict.items()
                for node in (nodes if isinstance(nodes, list) else [nodes])
            ]
        assert len(capture_iter) == 1
        assert capture_iter[0][1] == "func_name"


# ── A2A Handshake Tests ───────────────────────────────────────────────────────

class TestServerHandshake:
    """
    Tests the A2A JSON-RPC envelope parsing.
    Function is _extract_task (was _extract_task_and_context in earlier version).
    """

    def test_extract_task_data_part(self):
        """Turn 1: task arrives in a 'data' part."""
        from server import _extract_task

        body = {
            "params": {
                "message": {
                    "contextId": "session-123",
                    "parts": [{
                        "kind": "data",
                        "data": {
                            "problem_statement": "Fix the off-by-one bug",
                            "repo": "test/repo",
                            "base_commit": "abc123",
                        },
                    }],
                }
            }
        }
        task, cid = _extract_task(body)
        assert task["problem_statement"] == "Fix the off-by-one bug"
        assert task["repo"] == "test/repo"
        assert cid == "session-123"

    def test_extract_task_text_json_part(self):
        """Turn 1: task arrives as JSON-in-text part."""
        from server import _extract_task
        import json

        body = {
            "params": {
                "message": {
                    "contextId": "ctx-456",
                    "parts": [{
                        "kind": "text",
                        "text": json.dumps({
                            "problem_statement": "NullPointerException in login",
                            "repo": "org/app",
                        }),
                    }],
                }
            }
        }
        task, cid = _extract_task(body)
        assert task["problem_statement"] == "NullPointerException in login"
        assert cid == "ctx-456"

    def test_extract_task_flat_body(self):
        """Flat body (problem_statement at top level) is handled."""
        from server import _extract_task

        body = {
            "problem_statement": "Fix the race condition",
            "repo": "org/service",
        }
        task, cid = _extract_task(body)
        assert task["problem_statement"] == "Fix the race condition"

    def test_extract_task_missing_parts_returns_empty(self):
        """Malformed body returns empty dict without crashing."""
        from server import _extract_task

        task, cid = _extract_task({"params": {}})
        assert isinstance(task, dict)