"""
MCTS Engine
===========
Lightweight Monte-Carlo Tree Search for action selection.

Each node represents a state in the trajectory (codebase exploration).
UCT (Upper Confidence Bound for Trees) balances exploitation vs exploration:

    UCT(j) = V̄_j + c * sqrt( ln(N) / n_j )

where:
    V̄_j = average reward of node j
    N    = visit count of parent
    n_j  = visit count of node j
    c    = exploration constant (default √2)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("purple_agent.mcts")

EXPLORATION_CONSTANT = math.sqrt(2)


# ── Node ──────────────────────────────────────────────────────────────────────

@dataclass
class MCTSNode:
    state: dict                          # NodeState from state manager
    parent: MCTSNode | None = None
    action: dict | None = None           # action that led to this node
    children: list[MCTSNode] = field(default_factory=list)

    _visit_count: int = 0
    _value_sum: float = 0.0

    @property
    def visit_count(self) -> int:
        return self._visit_count

    @property
    def value(self) -> float:
        if self._visit_count == 0:
            return 0.0
        return self._value_sum / self._visit_count

    def uct_score(self, parent_visits: int, c: float = EXPLORATION_CONSTANT) -> float:
        if self._visit_count == 0:
            return float("inf")  # unvisited nodes get priority
        exploitation = self.value
        exploration = c * math.sqrt(math.log(parent_visits + 1) / self._visit_count)
        return exploitation + exploration

    def update(self, reward: float):
        self._visit_count += 1
        self._value_sum += reward

    def best_child(self) -> MCTSNode | None:
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.uct_score(self._visit_count))

    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ── Engine ────────────────────────────────────────────────────────────────────

class MCTSEngine:
    """
    Manages the MCTS tree across a single task's trajectory.

    Usage:
        engine = MCTSEngine(root_node, branches=3)

        # Get next action (with static scores from PRM)
        action, node = engine.select_action(candidates)

        # After executing and observing result:
        engine.backpropagate(reward)
    """

    def __init__(self, root: MCTSNode, branches: int = 3):
        self.root = root
        self.branches = branches
        self._current_node = root
        self._pending_child: MCTSNode | None = None

    def select_action(
        self,
        candidates: list[tuple[dict, float]],
    ) -> tuple[dict, MCTSNode]:
        """
        Given a list of (action, static_score) candidates from the LLM,
        expand the current node and select the best branch via UCT.

        Returns the chosen (action, node).
        """
        # Expand: create a child node for each candidate
        for action, static_score in candidates:
            child = MCTSNode(
                state=self._current_node.state.copy(),
                parent=self._current_node,
                action=action,
            )
            # Initialise with static PRM score (prior)
            child.update(static_score)
            self._current_node.children.append(child)

        # Select best child by UCT
        best = self._current_node.best_child()
        if best is None:
            # Fallback: just use first candidate
            action, score = candidates[0]
            return action, self._current_node

        self._pending_child = best
        logger.debug(
            "MCTS selected action=%s (UCT=%.3f, visits=%d, value=%.3f)",
            best.action.get("action"),
            best.uct_score(self._current_node.visit_count),
            best.visit_count,
            best.value,
        )
        return best.action, best

    def backpropagate(self, reward: float):
        """
        Update the pending child and all ancestors with the observed reward.
        Called after the green agent returns an observation.
        """
        node = self._pending_child or self._current_node
        while node is not None:
            node.update(reward)
            node = node.parent

        # Advance current node to the selected child
        if self._pending_child is not None:
            self._current_node = self._pending_child
            self._pending_child = None

    def get_best_path_actions(self) -> list[dict]:
        """Return the sequence of actions on the highest-value path from root."""
        path = []
        node = self.root
        while not node.is_leaf():
            best = max(node.children, key=lambda c: c.value)
            if best.action:
                path.append(best.action)
            node = best
        return path

    def stats(self) -> dict:
        return {
            "total_nodes": self._count_nodes(self.root),
            "current_depth": self._depth(self._current_node),
            "root_visits": self.root.visit_count,
            "root_value": self.root.value,
        }

    @staticmethod
    def _count_nodes(node: MCTSNode) -> int:
        return 1 + sum(MCTSEngine._count_nodes(c) for c in node.children)

    @staticmethod
    def _depth(node: MCTSNode) -> int:
        d = 0
        while node.parent is not None:
            d += 1
            node = node.parent
        return d
