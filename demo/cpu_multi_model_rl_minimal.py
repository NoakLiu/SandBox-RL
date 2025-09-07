#!/usr/bin/env python3
"""
CPU-only minimal multi-model RL demo using SandGraph core components.

- Uses MockLLM backend (no GPU / external models required)
- Creates a shared LLM manager and two logical models/agents
- Runs a tiny PPO-like loop through RLTrainer with synthetic rewards
"""

import time
import random
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is on sys.path for direct execution
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Prefer importing modules directly from core to avoid heavy package __init__ side-effects
CORE_DIR = Path(PROJECT_ROOT) / "sandgraph" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from llm_interface import create_llm_config, create_llm, SharedLLMManager  # type: ignore
from rl_algorithms import RLConfig, RLAlgorithm, RLTrainer  # type: ignore


# Lightweight task schema matching demo/multi_model_single_env_simple.py
class TrainingTask:
    def __init__(self, task_id: str, task_type: str, difficulty: float, reward_pool: float,
                 max_steps: int, required_models: int, cooperation_level: float):
        self.task_id = task_id
        self.task_type = task_type
        self.difficulty = difficulty
        self.reward_pool = reward_pool
        self.max_steps = max_steps
        self.required_models = required_models
        self.cooperation_level = cooperation_level


def build_shared_manager_mock() -> Any:
    # Build a MockLLM on CPU
    config = create_llm_config(backend="mock", model_name="mock_llm", device="cpu")
    llm = create_llm(config)
    return SharedLLMManager(llm)


def _generate_initial_tasks(num_tasks: int = 20) -> List[TrainingTask]:
    task_types = ["classification", "generation", "reasoning", "optimization", "collaboration"]
    tasks: List[TrainingTask] = []
    for i in range(num_tasks):
        tasks.append(
            TrainingTask(
                task_id=f"task_{i:03d}",
                task_type=random.choice(task_types),
                difficulty=random.uniform(0.3, 0.9),
                reward_pool=random.uniform(10.0, 100.0),
                max_steps=random.randint(5, 20),
                required_models=random.randint(2, 5),
                cooperation_level=random.uniform(0.0, 1.0),
            )
        )
    return tasks


def tiny_multi_model_cpu_run(episodes: int = 5, steps_per_episode: int = 8) -> Dict[str, Any]:
    shared = build_shared_manager_mock()

    # Register two logical models that share parameters
    shared.register_node("model_A", {"temperature": 0.6, "max_length": 128})
    shared.register_node("model_B", {"temperature": 0.8, "max_length": 128})

    # PPO config kept tiny for CPU
    rl_config = RLConfig(
        algorithm=RLAlgorithm.PPO,
        learning_rate=1e-3,
        batch_size=8,
        ppo_epochs=2,
        entropy_coef=0.01,
        value_loss_coef=0.5,
    )

    trainer = RLTrainer(rl_config, shared)

    tasks = _generate_initial_tasks()

    def build_task_prompt(task: TrainingTask, model_id: str, strategy: str) -> str:
        return (
            f"你正在参与多模型训练任务。\n"
            f"任务ID: {task.task_id}\n任务类型: {task.task_type}\n难度: {task.difficulty:.2f}\n"
            f"奖励池: {task.reward_pool:.2f}\n合作级别: {task.cooperation_level:.2f}\n"
            f"模型: {model_id}\n当前策略: {strategy}\n"
            f"请根据任务与合作级别，给出该步行动理由与简短计划。"
        )

    def step_joint(task: TrainingTask, t: int) -> Dict[str, Any]:
        # Action selection biased by cooperation level
        coop_bias = task.cooperation_level
        action_A = "cooperate" if random.random() < coop_bias else "compete"
        action_B = "cooperate" if random.random() < coop_bias else "compete"

        prompt_A = build_task_prompt(task, "model_A", action_A)
        prompt_B = build_task_prompt(task, "model_B", action_B)
        resp_A = shared.generate_for_node("model_A", prompt_A, reasoning_type="logical")
        resp_B = shared.generate_for_node("model_B", prompt_B, reasoning_type="logical")

        # Base rewards scale with confidence and lower difficulty
        scale = max(0.1, 1.0 - task.difficulty)
        base_A = resp_A.confidence * scale
        base_B = resp_B.confidence * scale

        # Cooperation/competition payoff shaped by cooperation_level
        payoff = 0.0
        if action_A == "cooperate" and action_B == "cooperate":
            payoff = 0.25 * task.cooperation_level
        elif action_A == "compete" and action_B == "compete":
            payoff = 0.25 * (1.0 - task.cooperation_level)
        else:
            payoff = -0.1

        reward_A = max(0.0, min(1.0, base_A + payoff))
        reward_B = max(0.0, min(1.0, base_B + payoff))

        done = (t == min(steps_per_episode, task.max_steps) - 1)
        state_A = {"t": t, "task": task.task_type, "coop": task.cooperation_level, "conf": resp_A.confidence}
        state_B = {"t": t, "task": task.task_type, "coop": task.cooperation_level, "conf": resp_B.confidence}

        return {
            "A": {"state": state_A, "action": action_A, "reward": reward_A, "done": done},
            "B": {"state": state_B, "action": action_B, "reward": reward_B, "done": done},
        }

    history = {"updates": [], "episodes": []}
    start = time.time()

    for ep in range(episodes):
        # Sample a task per episode similar to the demo environment
        task = random.choice(tasks)
        episode_reward_A = 0.0
        episode_reward_B = 0.0
        horizon = min(steps_per_episode, task.max_steps)
        for t in range(horizon):
            joint = step_joint(task, t)
            A = joint["A"]
            B = joint["B"]

            trainer.add_experience(A["state"], A["action"], A["reward"], A["done"], group_id="team")
            trainer.add_experience(B["state"], B["action"], B["reward"], B["done"], group_id="team")

            episode_reward_A += A["reward"]
            episode_reward_B += B["reward"]

            # Occasionally perform a small PPO update on shared parameters
            if (t + 1) % rl_config.batch_size == 0:
                upd = trainer.update_policy()
                history["updates"].append(upd)

        history["episodes"].append({
            "episode": ep,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "cooperation_level": round(task.cooperation_level, 2),
            "reward_A": round(episode_reward_A, 3),
            "reward_B": round(episode_reward_B, 3),
        })

    elapsed = time.time() - start
    return {"elapsed_sec": round(elapsed, 3), **history}


def main():
    result = tiny_multi_model_cpu_run(episodes=3, steps_per_episode=8)
    print("CPU Multi-Model RL (Mock) done.")
    print("Elapsed(s):", result["elapsed_sec"])
    for ep in result["episodes"]:
        print(f"Episode {ep['episode']}: A={ep['reward_A']}, B={ep['reward_B']}")
    if result["updates"]:
        last = result["updates"][-1]
        print("Last update status:", last.get("status"), "update_count:", last.get("trajectory_count", "-"))


if __name__ == "__main__":
    main()


