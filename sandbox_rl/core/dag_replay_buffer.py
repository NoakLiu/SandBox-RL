"""
Paper-aligned DAG Replay Buffer abstraction for RL.

Provides a light wrapper around existing trajectory buffers to
emphasize DAG-structured episodes, consistent with the paper.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class DAGTraceStep:
    node_id: str
    x: Any
    s: Any
    y: Any
    r: float


class DAGReplayBuffer:
    def __init__(self) -> None:
        self.episodes: List[List[DAGTraceStep]] = []

    def start_episode(self) -> None:
        self.episodes.append([])

    def add_step(self, step: DAGTraceStep) -> None:
        if not self.episodes:
            self.episodes.append([])
        self.episodes[-1].append(step)

    def finalize_episode(self) -> List[DAGTraceStep]:
        if not self.episodes:
            return []
        return self.episodes[-1]

    def get_all_episodes(self) -> List[List[DAGTraceStep]]:
        return self.episodes

    def clear(self) -> None:
        self.episodes.clear()


__all__ = [
    "DAGReplayBuffer",
    "DAGTraceStep",
]


