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
from typing import Dict, Any

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


def build_shared_manager_mock() -> Any:
    # Build a MockLLM on CPU
    config = create_llm_config(backend="mock", model_name="mock_llm", device="cpu")
    llm = create_llm(config)
    return SharedLLMManager(llm)


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

    def synthetic_env_step(model_id: str, t: int) -> Dict[str, Any]:
        # Query the shared LLM for a trivial prompt (MockLLM generates instantly)
        prompt = f"第{t}步: 请选择策略以提高准确率与效率，并给出简短理由。"
        resp = shared.generate_for_node(model_id, prompt, reasoning_type="logical")
        # Build a tiny reward: mix of confidence and some interaction term
        base_reward = resp.confidence
        shaping = 0.05 if "策略" in resp.text else 0.0
        reward = min(1.0, base_reward + shaping)
        action = "cooperate" if random.random() < 0.5 else "compete"
        done = (t == steps_per_episode - 1)
        state = {"t": t, "model_id": model_id, "conf": resp.confidence}
        return {"state": state, "action": action, "reward": reward, "done": done}

    history = {"updates": [], "episodes": []}
    start = time.time()

    for ep in range(episodes):
        episode_reward_A = 0.0
        episode_reward_B = 0.0
        for t in range(steps_per_episode):
            step_A = synthetic_env_step("model_A", t)
            step_B = synthetic_env_step("model_B", t)

            trainer.add_experience(step_A["state"], step_A["action"], step_A["reward"], step_A["done"], group_id="team")
            trainer.add_experience(step_B["state"], step_B["action"], step_B["reward"], step_B["done"], group_id="team")

            episode_reward_A += step_A["reward"]
            episode_reward_B += step_B["reward"]

            # Occasionally perform a small PPO update on shared parameters
            if (t + 1) % rl_config.batch_size == 0:
                upd = trainer.update_policy()
                history["updates"].append(upd)

        history["episodes"].append({
            "episode": ep,
            "reward_A": round(episode_reward_A, 3),
            "reward_B": round(episode_reward_B, 3)
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


