#!/usr/bin/env python3
import sys
from pathlib import Path
import random
import json
import matplotlib.pyplot as plt

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CORE_DIR = Path(PROJECT_ROOT) / "sandgraph" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from coop_compete_benchmark import SimplePG, OurMethodPolicy  # type: ignore
from maze_env import MazeEnv, MazeConfig  # type: ignore


def run_episode(env: MazeEnv, policy, eps: float = 0.1):
    pos = env.reset()
    traj = []
    done = False
    while not done:
        # map Bernoulli cooperate prob to 4-dir with simple heuristic
        if isinstance(policy, SimplePG) or isinstance(policy, OurMethodPolicy):
            a_bin, _ = policy.act()
            # stochastic exploration
            if random.random() < eps:
                a = random.randint(0, 3)
            else:
                # bias towards moving right/down when a_bin=1, else random
                a = random.choice([1, 2]) if a_bin == 1 else random.randint(0, 3)
        else:
            a = random.randint(0, 3)
        npos, r, done = env.step(a)
        traj.append(((pos, a), r))
        pos = npos
    return traj


def reinforce_update(policy, traj):
    # convert to (action_binary, reward) with simple shaping: right/down->1 else 0
    steps = []
    for (state_action, r) in traj:
        (_, a) = state_action
        a_bin = 1 if a in (1, 2) else 0
        steps.append((a_bin, r))
    rewards = [r for (_, r) in steps]
    baseline = sum(rewards) / len(rewards)
    if isinstance(policy, SimplePG):
        p = policy._sigmoid(policy.theta)
        grad = 0.0
        for (ab, r) in steps:
            grad += (ab - p) * (r - baseline)
        grad /= max(1, len(steps))
        policy.theta += policy.lr * grad
    elif isinstance(policy, OurMethodPolicy):
        p = policy._sigmoid(policy.theta)
        grad = 0.0
        for (ab, r) in steps:
            grad += (ab - p) * (r - baseline)
        grad /= max(1, len(steps))
        # use simple regime signal: progress towards goal assumed by right/down preference
        coop_regime = 0.7
        gain = (2 * coop_regime - 1.0) * (2 * policy.opp_coop_est - 1.0)
        policy.theta += policy.lr * grad * (1.0 + 0.5 * gain)


def main():
    cfg = MazeConfig(width=7, height=7, start=(0, 0), goal=(6, 6), walls=set(), max_steps=64)
    env = MazeEnv(cfg)
    episodes = 200
    pg = SimplePG(lr=0.05)
    our = OurMethodPolicy(lr=0.05, momentum=0.9)
    pg_returns = []
    our_returns = []

    for ep in range(episodes):
        traj_pg = run_episode(env, pg)
        pg_returns.append(sum(r for (_, r) in traj_pg))
        reinforce_update(pg, traj_pg)

        traj_our = run_episode(env, our)
        our_returns.append(sum(r for (_, r) in traj_our))
        # observe pseudo opponent coop as down/right ratio
        dr_ratio = sum(1 for ((_, a), _) in traj_our if a in (1, 2)) / max(1, len(traj_our))
        our.opp_coop_est = 0.9 * our.opp_coop_est + 0.1 * dr_ratio
        reinforce_update(our, traj_our)

    params = asdict(cfg)
    if isinstance(params.get("walls"), set):
        params["walls"] = list(params["walls"])  # JSON serializable
    out = {
        "params": params,
        "returns": {"PG": pg_returns, "OUR": our_returns}
    }
    out_path = Path("training_outputs") / "maze_pg_vs_our.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"JSON written: {out_path}")

    # Plot
    Path("visualization_outputs").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(pg_returns, label="PG")
    plt.plot(our_returns, label="OUR")
    plt.xlabel("Episode")
    plt.ylabel("Return (sum of rewards)")
    plt.title("Maze PG vs OUR")
    plt.legend()
    png = Path("visualization_outputs") / "maze_pg_vs_our.png"
    plt.tight_layout()
    plt.savefig(png, dpi=150)
    plt.close()
    print(f"Plot written: {png}")


if __name__ == "__main__":
    from dataclasses import asdict
    main()


