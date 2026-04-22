"""
Node State Manager
==================
Each MCTS node stores its own "branch state" so that when the agent
jumps between branches, context is swapped correctly.

State per node:
  working_set    – list of currently open/explored files
  discovery_log  – facts discovered (file:line -> description)
  current_patch  – cumulative diff applied so far
  cwd            – current working directory
  turn           – turn counter within this branch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import SWETask


@dataclass
class NodeState:
    """Mutable state carried by each MCTS node."""

    cwd: str = "/workspace/repo"
    working_set: list[str] = field(default_factory=list)
    discovery_log: dict[str, str] = field(default_factory=dict)
    current_patch: str = ""
    turn: int = 0
    branch_id: str = "root"

    def copy(self) -> NodeState:
        return NodeState(
            cwd=self.cwd,
            working_set=list(self.working_set),
            discovery_log=dict(self.discovery_log),
            current_patch=self.current_patch,
            turn=self.turn,
            branch_id=self.branch_id,
        )

    def add_file(self, path: str):
        if path not in self.working_set:
            self.working_set.append(path)

    def add_discovery(self, location: str, fact: str):
        self.discovery_log[location] = fact

    def summarize(self) -> str:
        """Return a compact text summary for injection into the LLM context."""
        lines = []

        if self.working_set:
            lines.append("Open files: " + ", ".join(self.working_set[-5:]))

        if self.discovery_log:
            lines.append("Discoveries:")
            for loc, fact in list(self.discovery_log.items())[-8:]:
                lines.append(f"  • {loc}: {fact}")

        if self.current_patch:
            patch_lines = self.current_patch.strip().splitlines()
            lines.append(f"Current patch: {len(patch_lines)} lines")

        return "\n".join(lines) if lines else "No state yet."


class NodeStateManager:
    """Factory and update helper for NodeState objects."""

    def new_root_state(self, task: SWETask) -> NodeState:
        return NodeState(
            cwd=task.cwd,
            branch_id="root",
        )

    def update_from_observation(
        self,
        state: NodeState,
        action: dict,
        obs: dict,
    ) -> NodeState:
        """
        Merge observation data into state (in-place mutation of a copy).
        """
        new_state = state.copy()
        new_state.turn += 1
        new_state.cwd = obs.get("cwd", state.cwd)

        stdout = obs.get("stdout", "")

        # Extract file paths mentioned in stdout
        import re
        files = re.findall(r'([\w/.-]+\.py)(?::(\d+))?', stdout)
        for fpath, lineno in files[:10]:  # cap at 10
            new_state.add_file(fpath)
            if lineno:
                new_state.add_discovery(
                    f"{fpath}:{lineno}",
                    f"Referenced in output of: {action.get('content', '')[:50]}",
                )

        # Track patch accumulation
        if action.get("action") == "patch":
            new_state.current_patch = action.get("content", "")

        return new_state
