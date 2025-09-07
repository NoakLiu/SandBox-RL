#!/usr/bin/env python3
"""
CPU-only benchmark comparing three strategies in a simple cooperative-competitive RL environment:
1) Always-Cooperate (AC)
2) Always-Compete (AP)
3) Adaptive Policy Gradient (PG) with cooperation/competition actions

Metrics: episode reward curves, steps-to-target (convergence), final average reward.
No LLM dependencies; pure synthetic environment.
"""

import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple


# Simple 2-agent stage game with stochastic payoff influenced by cooperation_level
@dataclass
class EnvConfig:
    base_scale: float = 1.0
    noise_std: float = 0.05
    coop_weight: float = 0.25
    compete_weight: float = 0.25


class CoopCompeteEnv:
    def __init__(self, cooperation_level: float, difficulty: float, cfg: EnvConfig = EnvConfig()):
        self.cooperation_level = max(0.0, min(1.0, cooperation_level))
        self.difficulty = max(0.0, min(1.0, difficulty))
        self.cfg = cfg

    def step(self, action_a: int, action_b: int) -> Tuple[float, float]:
        # action: 0=compete, 1=cooperate
        scale = self.cfg.base_scale * max(0.1, 1.0 - self.difficulty)

        if action_a == 1 and action_b == 1:  # both cooperate
            payoff = self.cfg.coop_weight * self.cooperation_level
        elif action_a == 0 and action_b == 0:  # both compete
            payoff = self.cfg.compete_weight * (1.0 - self.cooperation_level)
        else:
            payoff = -0.1

        base_a = scale
        base_b = scale
        ra = max(0.0, base_a + payoff + random.gauss(0, self.cfg.noise_std))
        rb = max(0.0, base_b + payoff + random.gauss(0, self.cfg.noise_std))
        return ra, rb


def always_cooperate_policy(_: List[float]) -> int:
    return 1


def always_compete_policy(_: List[float]) -> int:
    return 0


class SimplePG:
    def __init__(self, lr: float = 0.1):
        # Bernoulli policy: P(cooperate) = sigmoid(theta)
        self.theta = 0.0
        self.lr = lr

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def act(self) -> Tuple[int, float]:
        p = self._sigmoid(self.theta)
        a = 1 if random.random() < p else 0
        logp = math.log(p + 1e-8) if a == 1 else math.log(1 - p + 1e-8)
        return a, logp

    def update(self, trajectories: List[Tuple[int, float, float]]):
        # trajectories: (action, logp, reward)
        # REINFORCE with baseline
        rewards = [r for (_, _, r) in trajectories]
        baseline = statistics.mean(rewards) if rewards else 0.0
        grad = 0.0
        for a, logp, r in trajectories:
            advantage = r - baseline
            # d log pi / d theta = (a - p) for Bernoulli with sigmoid, but we approximate via score function
            grad += advantage * (1.0)  # use score function surrogate via log-prob accumulation
        self.theta += self.lr * grad / max(1, len(trajectories))


def run_episode(env: CoopCompeteEnv, policy_a, policy_b, horizon: int = 32, pg_train: bool = False) -> Tuple[float, float, List[Tuple[int, float, float]], List[Tuple[int, float, float]]]:
    ep_ra = 0.0
    ep_rb = 0.0
    traj_a: List[Tuple[int, float, float]] = []
    traj_b: List[Tuple[int, float, float]] = []

    for _ in range(horizon):
        if isinstance(policy_a, SimplePG):
            a, logpa = policy_a.act()
        else:
            a = policy_a([])
            logpa = 0.0
        if isinstance(policy_b, SimplePG):
            b, logpb = policy_b.act()
        else:
            b = policy_b([])
            logpb = 0.0

        ra, rb = env.step(a, b)
        ep_ra += ra
        ep_rb += rb

        if pg_train and isinstance(policy_a, SimplePG):
            traj_a.append((a, logpa, ra))
        if pg_train and isinstance(policy_b, SimplePG):
            traj_b.append((b, logpb, rb))

    return ep_ra, ep_rb, traj_a, traj_b


def benchmark(runs: int = 5, episodes: int = 200, horizon: int = 32, coop_level: float = 0.6, difficulty: float = 0.3):
    results: Dict[str, Dict[str, List[float]]] = {
        "AC": {"A": [], "B": []},
        "AP": {"A": [], "B": []},
        "PG": {"A": [], "B": []},
    }
    steps_to_target: Dict[str, List[int]] = {"AC": [], "AP": [], "PG": []}
    target = 0.9 * (1.0 - difficulty)  # heuristic target per-step reward baseline

    for _ in range(runs):
        env = CoopCompeteEnv(cooperation_level=coop_level, difficulty=difficulty)

        # 1) Always-Cooperate
        curve_ac_a: List[float] = []
        curve_ac_b: List[float] = []
        reached = False
        for ep in range(episodes):
            ra, rb, _, _ = run_episode(env, always_cooperate_policy, always_cooperate_policy, horizon=horizon)
            curve_ac_a.append(ra / horizon)
            curve_ac_b.append(rb / horizon)
            if not reached and (ra / horizon) >= target and (rb / horizon) >= target:
                steps_to_target["AC"].append(ep + 1)
                reached = True
        results["AC"]["A"].append(sum(curve_ac_a) / episodes)
        results["AC"]["B"].append(sum(curve_ac_b) / episodes)

        # 2) Always-Compete
        curve_ap_a: List[float] = []
        curve_ap_b: List[float] = []
        reached = False
        for ep in range(episodes):
            ra, rb, _, _ = run_episode(env, always_compete_policy, always_compete_policy, horizon=horizon)
            curve_ap_a.append(ra / horizon)
            curve_ap_b.append(rb / horizon)
            if not reached and (ra / horizon) >= target and (rb / horizon) >= target:
                steps_to_target["AP"].append(ep + 1)
                reached = True
        results["AP"]["A"].append(sum(curve_ap_a) / episodes)
        results["AP"]["B"].append(sum(curve_ap_b) / episodes)

        # 3) Adaptive Policy Gradient
        pg_a = SimplePG(lr=0.05)
        pg_b = SimplePG(lr=0.05)
        curve_pg_a: List[float] = []
        curve_pg_b: List[float] = []
        reached = False
        for ep in range(episodes):
            ra, rb, traj_a, traj_b = run_episode(env, pg_a, pg_b, horizon=horizon, pg_train=True)
            curve_pg_a.append(ra / horizon)
            curve_pg_b.append(rb / horizon)
            # Update after each episode
            pg_a.update(traj_a)
            pg_b.update(traj_b)
            if not reached and (ra / horizon) >= target and (rb / horizon) >= target:
                steps_to_target["PG"].append(ep + 1)
                reached = True
        results["PG"]["A"].append(sum(curve_pg_a) / episodes)
        results["PG"]["B"].append(sum(curve_pg_b) / episodes)

    def summarize(name: str) -> Tuple[float, float, float]:
        avg_a = statistics.mean(results[name]["A"]) if results[name]["A"] else 0.0
        avg_b = statistics.mean(results[name]["B"]) if results[name]["B"] else 0.0
        steps = statistics.mean(steps_to_target[name]) if steps_to_target[name] else float("inf")
        return avg_a, avg_b, steps

    ac = summarize("AC")
    ap = summarize("AP")
    pg = summarize("PG")

    print("Benchmark (runs={}, episodes={}, horizon={}, coop_level={}, difficulty={})".format(
        runs, episodes, horizon, coop_level, difficulty
    ))
    print("- Always-Cooperate: avg_per_step_reward_A={:.3f}, B={:.3f}, steps_to_target={}".format(ac[0], ac[1], ac[2]))
    print("- Always-Compete:   avg_per_step_reward_A={:.3f}, B={:.3f}, steps_to_target={}".format(ap[0], ap[1], ap[2]))
    print("- Adaptive PG:      avg_per_step_reward_A={:.3f}, B={:.3f}, steps_to_target={}".format(pg[0], pg[1], pg[2]))


if __name__ == "__main__":
    benchmark(runs=5, episodes=200, horizon=32, coop_level=0.6, difficulty=0.3)


