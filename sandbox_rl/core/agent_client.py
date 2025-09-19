#!/usr/bin/env python3
"""
Agent client adapter protocol and a minimal local client.
"""

from typing import Protocol, Dict, Any
from .trainer_server import TrainerServer, Sample, Result
from .trajectory import Trajectory, TrajectoryStep


class AgentAdapter(Protocol):
    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def learn(self, trajectory: Trajectory) -> None:
        ...


class LocalAgentClient:
    def __init__(self, agent_id: str, adapter: AgentAdapter, server: TrainerServer):
        self.agent_id = agent_id
        self.adapter = adapter
        self.server = server

    def run_once(self) -> bool:
        sample = self.server.get_sample()
        if sample is None:
            return False
        obs = sample.payload
        action_dict = self.adapter.act(obs)
        # Minimal trajectory: single-step
        step = TrajectoryStep(state=obs, action=action_dict, reward=0.0, done=True, info={})
        traj = Trajectory(agent_id=self.agent_id, episode_id=sample.sample_id, steps=[step])
        self.adapter.learn(traj)
        self.server.put_result(Result(sample_id=sample.sample_id, agent_id=self.agent_id, trajectory=traj.to_dict()))
        return True


