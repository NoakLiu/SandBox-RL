#!/usr/bin/env python3
"""
Standardized trajectory data structures and helpers.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json


@dataclass
class TrajectoryStep:
    state: Dict[str, Any]
    action: Any
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    agent_id: str
    episode_id: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "episode_id": self.episode_id,
            "steps": [asdict(s) for s in self.steps],
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Trajectory":
        steps = [TrajectoryStep(**s) for s in d.get("steps", [])]
        return Trajectory(agent_id=d["agent_id"], episode_id=d["episode_id"], steps=steps, metadata=d.get("metadata", {}))


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


