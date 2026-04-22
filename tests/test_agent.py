"""
Tests for Purple Agent Components
==================================
Run with: pytest tests/ -v

These tests mock the LLM so no GPU is needed.
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_task_message():
    return {
        "problem_statement": (
            "The `add_to_set` function in utils/collections.py crashes "
            "when given None values in the input list. It should treat "
            "None as a regular hashable value."
        ),
        "cwd": "/workspace/repo",
        "hints_text": "Look at utils/collections.py around line 42.",
        "python_version": "3.10",
        "fail_to_pass": ["tests/test_collections.py::test_add_none_to_set"],
    }


@pytest.fixture
def mock_llm_bash_response():
    return """<thought>
Let me start by understanding the repository structure and finding the relevant file.
I need to look at utils/collections.py and understand the add_to_set function.
</thought>
<command>bash</command>
<content>
find /workspace/repo -name "collections.py" | head -5 && cat -n /workspace/repo/utils/collections.py
</content>"""


@pytest.fixture
def mock_llm_patch_response():
    return """<thought>
I found the bug. On line 42, the function does `if item:` which is falsy for None.
It should be `if item is not None:` or just iterate without the guard.
The fix is to remove the None check and let the set handle it.
</thought>
<command>patch</command>
<content>
diff --git a/utils/collections.py b/utils/collections.py
--- a/utils/collections.py
+++ b/utils/collections.py
@@ -39,7 +39,7 @@ def add_to_set(items, existing=None):
     result = existing or set()
     for item in items:
-        if item:
+        if item is not None:
             result.add(item)
     return result
</content>"""


# ── MCTS Tests ────────────────────────────────────────────────────────────────

class TestMCTSNode:
    def test_uct_score_unvisited(self):
        from mcts import MCTSNode
        node = MCTSNode(state={})
        assert node.uct_score(10) == float("inf")

    def test_uct_score_visited(self):
        from mcts import MCTSNode
        import math
        node = MCTSNode(state={})
        node.update(0.8)
        node.update(0.6)
        # value = 0.7, UCT > 0.7 due to exploration term
        score = node.uct_score(10)
        assert score > 0.7

    def test_best_child_selection(self):
        from mcts import MCTSNode
        root = MCTSNode(state={})
        root.update(0.5)

        c1 = MCTSNode(state={}, parent=root)
        c1.update(0.9)

        c2 = MCTSNode(state={}, parent=root)
        c2.update(0.3)

        root.children = [c1, c2]
        # c2 unvisited? No — both visited. c1 has higher value.
        best = root.best_child()
        assert best is not None


class TestMCTSEngine:
    def test_select_action_returns_candidate(self):
        from mcts import MCTSEngine, MCTSNode
        root = MCTSNode(state={})
        engine = MCTSEngine(root, branches=3)

        candidates = [
            ({"action": "bash", "content": "find . -name '*.py'"}, 0.7),
            ({"action": "bash", "content": "cat file.py"}, 0.5),
        ]
        action, node = engine.select_action(candidates)
        assert action["action"] in ("bash", "debug", "patch")

    def test_backpropagate_updates_root(self):
        from mcts import MCTSEngine, MCTSNode
        root = MCTSNode(state={})
        engine = MCTSEngine(root, branches=2)

        candidates = [({"action": "bash", "content": "ls"}, 0.5)]
        engine.select_action(candidates)
        engine.backpropagate(0.8)

        assert root.visit_count > 0


# ── PRM Tests ─────────────────────────────────────────────────────────────────

class TestProgrammablePRM:
    def setup_method(self):
        from prm import ProgrammablePRM
        from agent import SWETask
        self.prm = ProgrammablePRM()
        self.task = SWETask(
            problem_statement="Bug in utils/collections.py add_to_set function with None",
            fail_to_pass=["tests/test_collections.py::test_add_none"],
        )

    def test_format_reward_valid_bash(self):
        action = {"action": "bash", "content": "cat -n utils/collections.py"}
        score = self.prm.score_static(action, self.task)
        assert 0.0 <= score <= 1.0
        assert score > 0.3

    def test_format_reward_empty_content(self):
        action = {"action": "bash", "content": ""}
        score = self.prm.score_static(action, self.task)
        assert score < 0.3

    def test_format_reward_valid_patch(self):
        action = {
            "action": "patch",
            "content": (
                "diff --git a/utils/collections.py b/utils/collections.py\n"
                "--- a/utils/collections.py\n"
                "+++ b/utils/collections.py\n"
                "@@ -42,1 +42,1 @@\n"
                "-        if item:\n"
                "+        if item is not None:\n"
            ),
        }
        score = self.prm.score_static(action, self.task)
        assert score > 0.4

    def test_observation_reward_tests_passed(self):
        obs = {
            "stdout": "test_add_none_to_set PASSED\n1 passed in 0.12s",
            "stderr": "",
            "cwd": "/workspace/repo",
        }
        score = self.prm.score_observation(obs, self.task)
        assert score > 0.7

    def test_observation_reward_errors(self):
        obs = {
            "stdout": "",
            "stderr": "Traceback (most recent call last):\n  File ...\nTypeError: ...",
            "cwd": "/workspace/repo",
        }
        score = self.prm.score_observation(obs, self.task)
        assert score < 0.7


# ── State Manager Tests ───────────────────────────────────────────────────────

class TestNodeStateManager:
    def test_new_root_state(self):
        from state import NodeStateManager
        from agent import SWETask
        mgr = NodeStateManager()
        task = SWETask(problem_statement="test", cwd="/workspace/repo")
        state = mgr.new_root_state(task)
        assert state.cwd == "/workspace/repo"
        assert state.working_set == []
        assert state.discovery_log == {}

    def test_state_copy_is_independent(self):
        from state import NodeState
        s = NodeState(cwd="/a", working_set=["file.py"])
        s2 = s.copy()
        s2.working_set.append("other.py")
        assert "other.py" not in s.working_set

    def test_update_from_observation_extracts_files(self):
        from state import NodeStateManager
        from agent import SWETask
        mgr = NodeStateManager()
        task = SWETask(problem_statement="", cwd="/workspace")
        state = mgr.new_root_state(task)

        obs = {
            "stdout": "utils/collections.py:42: found issue\nutils/helpers.py:10: related",
            "stderr": "",
            "cwd": "/workspace/repo",
        }
        action = {"action": "bash", "content": "grep -r 'add_to_set' ."}
        new_state = mgr.update_from_observation(state, action, obs)
        assert len(new_state.working_set) > 0
        assert new_state.cwd == "/workspace/repo"


# ── Agent Integration Test ────────────────────────────────────────────────────

class TestPurpleAgent:
    def test_agent_responds_with_valid_action(self, sample_task_message, mock_llm_bash_response):
        """Agent should return a valid A2A action dict."""
        from agent import PurpleAgent

        with patch("llm_client.LLMClient.complete", return_value=mock_llm_bash_response):
            agent = PurpleAgent(
                model_base_url="http://localhost:8000",
                use_mcts=False,  # simpler path for unit test
            )
            action = agent.respond(sample_task_message)

        assert "action" in action
        assert action["action"] in ("bash", "debug", "patch")
        assert "content" in action

    def test_agent_parses_patch_action(self, sample_task_message, mock_llm_patch_response):
        """Agent should correctly parse a patch response."""
        from agent import PurpleAgent

        with patch("llm_client.LLMClient.complete", return_value=mock_llm_patch_response):
            agent = PurpleAgent(
                model_base_url="http://localhost:8000",
                use_mcts=False,
            )
            action = agent.respond(sample_task_message)

        assert action["action"] == "patch"
        assert "diff --git" in action["content"]

    def test_agent_force_patch_near_turn_limit(self, mock_llm_patch_response):
        """Agent should force a patch when nearing max_turns."""
        from agent import PurpleAgent

        with patch("llm_client.LLMClient.complete", return_value=mock_llm_patch_response):
            agent = PurpleAgent(
                model_base_url="http://localhost:8000",
                max_turns=5,
                use_mcts=False,
            )
            msg = {
                "problem_statement": "Fix the bug",
                "cwd": "/workspace/repo",
            }
            # Advance to near the turn limit
            session_id = str(abs(hash("Fix the bug"[:100])))
            agent.respond(msg)
            session = agent._sessions[session_id]
            session["turn"] = 4  # one before max_turns - 2 + 1

            obs_msg = {
                "session_id": session_id,
                "stdout": "some output",
                "stderr": "",
                "cwd": "/workspace/repo",
            }
            action = agent.respond(obs_msg)
            # Should produce a patch
            assert action["action"] == "patch"

    def test_session_persistence(self, sample_task_message):
        """Second call with session_id should reuse existing session."""
        from agent import PurpleAgent

        bash_resp = "<thought>explore</thought><command>bash</command><content>ls</content>"

        with patch("llm_client.LLMClient.complete", return_value=bash_resp):
            agent = PurpleAgent(
                model_base_url="http://localhost:8000",
                use_mcts=False,
            )
            # First call
            action1 = agent.respond(sample_task_message)
            assert len(agent._sessions) == 1

            session_id = list(agent._sessions.keys())[0]

            # Second call with session_id + observation
            obs_msg = {
                "session_id": session_id,
                "stdout": "utils/collections.py found",
                "stderr": "",
                "cwd": "/workspace/repo",
            }
            action2 = agent.respond(obs_msg)
            # Still only one session
            assert len(agent._sessions) == 1


# ── Server Tests ──────────────────────────────────────────────────────────────

class TestServer:
    @pytest.mark.asyncio
    async def test_agent_card_endpoint(self):
        from httpx import AsyncClient, ASGITransport
        from server import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/.well-known/agent.json")
            assert r.status_code == 200
            data = r.json()
            assert "name" in data
            assert "skills" in data

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        from httpx import AsyncClient, ASGITransport
        from server import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_task_endpoint(self, sample_task_message):
        bash_resp = "<thought>start</thought><command>bash</command><content>find . -name '*.py'</content>"

        from httpx import AsyncClient, ASGITransport
        from server import app, _agent
        import server

        mock_agent = MagicMock()
        mock_agent.respond.return_value = {"action": "bash", "content": "find . -name '*.py'"}

        original = server._agent
        server._agent = mock_agent
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                r = await client.post("/", json=sample_task_message)
                assert r.status_code == 200
                data = r.json()
                assert "result" in data
        finally:
            server._agent = original
