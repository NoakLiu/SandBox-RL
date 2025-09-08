"""
Paper-aligned RL Engine wrapper exposing PPO/GRPO over DAG traces.

This wraps existing trainers to provide a simple interface consistent
with the paper: construct engine with an LLM manager and update via
PPO/GRPO using episodes from a DAG replay buffer.
"""

from typing import Dict, Any

from .rl_algorithms import RLAlgorithm, create_ppo_trainer, create_grpo_trainer


class RLEngine:
    def __init__(self, algorithm: str = "ppo", learning_rate: float = 2e-4):
        self.algorithm = algorithm.lower()
        self.learning_rate = learning_rate
        self.trainer = None

    def initialize(self, llm_manager) -> None:
        if self.algorithm == "ppo":
            self.trainer = create_ppo_trainer(llm_manager, self.learning_rate)
        elif self.algorithm == "grpo":
            self.trainer = create_grpo_trainer(llm_manager, self.learning_rate)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def add_step(self, step: Dict[str, Any], group_id: str = "default") -> None:
        if not self.trainer:
            raise RuntimeError("RLEngine is not initialized. Call initialize().")
        self.trainer.add_trajectory_step(step, group_id)

    def update(self) -> Dict[str, Any]:
        if not self.trainer:
            raise RuntimeError("RLEngine is not initialized. Call initialize().")
        return self.trainer.update_policy(None)


__all__ = ["RLEngine"]


