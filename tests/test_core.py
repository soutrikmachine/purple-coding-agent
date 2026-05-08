import sys
import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

# Ensure src is in the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.llm_client import LLMClient
from core.docker_bridge import DockerBridge
from tools.test_engine import TestEngine
from tools.ast_graph import ASTGraphBuilder

# ── LLM Client Tests ─────────────────────────────────────────────────────────

class TestLLMClient:
    @pytest.fixture
    def client(self):
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
            return LLMClient()

    def test_parse_response_bash(self, client):
        """Verify extraction of thought and bash actions."""
        raw = "<thought>I will list files.</thought><action type=\"bash\">ls -R</action>"
        thought, act_type, content = client.parse_response(raw)
        assert thought == "I will list files."
        assert act_type == "bash"
        assert content == "ls -R"

    def test_parse_response_fallback(self, client):
        """Verify fallback when the model forgets the type attribute."""
        raw = "<thought>Check logs</thought><action>cat error.log</action>"
        thought, act_type, content = client.parse_response(raw)
        assert act_type == "bash"
        assert content == "cat error.log"

    def test_format_observation(self, client):
        """Verify XML wrapping of environment feedback."""
        obs = client.format_observation("File not found", 1)
        assert "<observation status=\"FAILED (Exit 1)\">" in obs
        assert "File not found" in obs


# ── Docker Bridge Tests ──────────────────────────────────────────────────────

class TestDockerBridge:
    @patch('docker.from_env')
    def test_execute_command_formatting(self, mock_docker):
        """Verify commands are wrapped in shell-level timeouts for safety."""
        bridge = DockerBridge(image_name="test-image")
        bridge.container = MagicMock()
        bridge.container.exec_run.return_value = (0, (b"hello", b""))
        
        code, output = bridge.execute_command("echo hello", timeout=30)
        
        # Check if the internal Docker call was made correctly with demux
        args, kwargs = bridge.container.exec_run.call_args
        assert "timeout 30s" in kwargs['cmd'][2]
        assert kwargs['demux'] is True
        assert output == "hello"


# ── Test Engine (Smarter Gate) Tests ─────────────────────────────────────────

class TestTestEngine:
    @pytest.fixture
    def engine(self):
        mock_bridge = MagicMock()
        return TestEngine(mock_bridge)

    def test_failure_parsing_pytest(self, engine):
        """Verify extraction of Pytest failure IDs."""
        output = "FAIL tests/test_api.py::test_login_auth - AssertionError"
        failures = engine._parse_failures(output)
        assert "tests/test_api.py::test_login_auth" in failures

    def test_gate_logic_baseline_delta(self, engine):
        """Verify the 'Relative Reward Signal' logic."""
        engine.baseline_failures = {"tests/old_broken.py::test_init"}
        
        # Scenario: Agent fixes the baseline but breaks nothing new
        engine.docker.execute_command.return_value = (0, "All tests passed")
        passed, msg = engine.run_test_gate()
        assert passed is True
        
        # Scenario: Agent introduces a NEW failure
        new_fail_out = "FAIL tests/new_bug.py::test_logic"
        engine.docker.execute_command.return_value = (1, new_fail_out)
        passed, msg = engine.run_test_gate()
        assert passed is False
        assert "tests/new_bug.py::test_logic" in msg


# ── AST Graph (Graph RAG) Tests ──────────────────────────────────────────────

class TestASTGraph:
    @patch('os.walk')
    def test_repo_skeleton_generation(self, mock_walk):
        """Verify the structural map generation for context priming."""
        mock_walk.return_value = [
            ('/workspace', ['src'], ['README.md']),
            ('/workspace/src', [], ['core.py']),
        ]
        
        builder = ASTGraphBuilder(repo_path="/workspace")
        
        # Mock file reading to avoid real tree-sitter overhead in unit tests
        with patch("tools.ast_graph.ASTGraphBuilder.parse_file_skeleton") as mock_parse:
            mock_parse.return_value = {"classes": ["Agent"], "functions": ["run"]}
            graph = builder.build_repo_graph()
            
            assert "File: `src/core.py`" in graph
            assert "Agent" in graph
            assert "run" in graph


# ── A2A Handshake Tests ──────────────────────────────────────────────────────

class TestServerHandshake:
    def test_extract_task_turn_1(self):
        """Verify parsing of the initial AgentBeats JSON-RPC payload."""
        from server import _extract_task_and_context
        
        mock_body = {
            "params": {
                "message": {
                    "contextId": "session-123",
                    "parts": [{"kind": "data", "data": {"problem_statement": "Fix bug", "repo": "test/repo"}}]
                }
            }
        }
        task, cid = _extract_task_and_context(mock_body)
        assert task["problem_statement"] == "Fix bug"
        assert cid == "session-123"