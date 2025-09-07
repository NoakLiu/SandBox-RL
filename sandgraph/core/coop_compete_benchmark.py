#!/usr/bin/env python3
"""
Core cooperative-competitive RL benchmark utilities (CPU-only, no LLM).

Provides:
- EnvConfig, CoopCompeteEnv: simple 2-agent payoff environment
- SimplePG: minimal policy-gradient agent over {compete, cooperate}
- run_benchmark: compare AC/AP/PG under given parameters and collect metrics
- write_report: persist results to JSON
"""

import math
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


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
        # 0=compete, 1=cooperate
        scale = self.cfg.base_scale * max(0.1, 1.0 - self.difficulty)
        if action_a == 1 and action_b == 1:
            payoff = self.cfg.coop_weight * self.cooperation_level
        elif action_a == 0 and action_b == 0:
            payoff = self.cfg.compete_weight * (1.0 - self.cooperation_level)
        else:
            payoff = -0.1
        ra = max(0.0, scale + payoff + random.gauss(0, self.cfg.noise_std))
        rb = max(0.0, scale + payoff + random.gauss(0, self.cfg.noise_std))
        return ra, rb


def always_cooperate_policy(_: List[float]) -> int:
    return 1


def always_compete_policy(_: List[float]) -> int:
    return 0


class SimplePG:
    def __init__(self, lr: float = 0.05):
        self.theta = 0.0  # Bernoulli logit
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
        if not trajectories:
            return
        rewards = [r for (_, _, r) in trajectories]
        baseline = statistics.mean(rewards)
        # Score function surrogate update with baseline
        grad = sum((r - baseline) for (_, _, r) in trajectories) / len(trajectories)
        self.theta += self.lr * grad


def _run_episode(env: CoopCompeteEnv, policy_a, policy_b, horizon: int, pg_train: bool) -> Tuple[float, float, List[Tuple[int, float, float]], List[Tuple[int, float, float]]]:
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


def run_benchmark(runs: int = 5, episodes: int = 200, horizon: int = 32, coop_level: float = 0.6, difficulty: float = 0.3, seed: int = 42) -> Dict[str, dict]:
    random.seed(seed)
    results: Dict[str, Dict[str, List[float]]] = {
        "AC": {"A": [], "B": []},
        "AP": {"A": [], "B": []},
        "PG": {"A": [], "B": []},
    }
    steps_to_target: Dict[str, List[int]] = {"AC": [], "AP": [], "PG": []}
    target = 0.9 * (1.0 - difficulty)

    for _ in range(runs):
        env = CoopCompeteEnv(cooperation_level=coop_level, difficulty=difficulty)

        # AC
        curve_ac_a: List[float] = []
        curve_ac_b: List[float] = []
        reached = False
        for ep in range(episodes):
            ra, rb, _, _ = _run_episode(env, always_cooperate_policy, always_cooperate_policy, horizon, False)
            ra_s = ra / horizon
            rb_s = rb / horizon
            curve_ac_a.append(ra_s)
            curve_ac_b.append(rb_s)
            if not reached and ra_s >= target and rb_s >= target:
                steps_to_target["AC"].append(ep + 1)
                reached = True
        results["AC"]["A"].append(sum(curve_ac_a) / episodes)
        results["AC"]["B"].append(sum(curve_ac_b) / episodes)

        # AP
        curve_ap_a: List[float] = []
        curve_ap_b: List[float] = []
        reached = False
        for ep in range(episodes):
            ra, rb, _, _ = _run_episode(env, always_compete_policy, always_compete_policy, horizon, False)
            ra_s = ra / horizon
            rb_s = rb / horizon
            curve_ap_a.append(ra_s)
            curve_ap_b.append(rb_s)
            if not reached and ra_s >= target and rb_s >= target:
                steps_to_target["AP"].append(ep + 1)
                reached = True
        results["AP"]["A"].append(sum(curve_ap_a) / episodes)
        results["AP"]["B"].append(sum(curve_ap_b) / episodes)

        # PG
        pg_a = SimplePG(lr=0.05)
        pg_b = SimplePG(lr=0.05)
        curve_pg_a: List[float] = []
        curve_pg_b: List[float] = []
        reached = False
        for ep in range(episodes):
            ra, rb, traj_a, traj_b = _run_episode(env, pg_a, pg_b, horizon, True)
            ra_s = ra / horizon
            rb_s = rb / horizon
            curve_pg_a.append(ra_s)
            curve_pg_b.append(rb_s)
            pg_a.update(traj_a)
            pg_b.update(traj_b)
            if not reached and ra_s >= target and rb_s >= target:
                steps_to_target["PG"].append(ep + 1)
                reached = True
        results["PG"]["A"].append(sum(curve_pg_a) / episodes)
        results["PG"]["B"].append(sum(curve_pg_b) / episodes)

    def summarize(name: str) -> Tuple[float, float, float, float, float]:
        a_list = results[name]["A"]
        b_list = results[name]["B"]
        avg_a = statistics.mean(a_list) if a_list else 0.0
        avg_b = statistics.mean(b_list) if b_list else 0.0
        med_steps = statistics.median(steps_to_target[name]) if steps_to_target[name] else float("inf")
        mean_steps = statistics.mean(steps_to_target[name]) if steps_to_target[name] else float("inf")
        std_steps = statistics.pstdev(steps_to_target[name]) if len(steps_to_target[name]) > 1 else 0.0
        return avg_a, avg_b, med_steps, mean_steps, std_steps

    ac = summarize("AC")
    ap = summarize("AP")
    pg = summarize("PG")

    return {
        "params": {
            "runs": runs,
            "episodes": episodes,
            "horizon": horizon,
            "coop_level": coop_level,
            "difficulty": difficulty,
            "target_per_step": target,
            "seed": seed,
        },
        "metrics": {
            "AC": {"avg_A": ac[0], "avg_B": ac[1], "median_steps": ac[2], "mean_steps": ac[3], "std_steps": ac[4]},
            "AP": {"avg_A": ap[0], "avg_B": ap[1], "median_steps": ap[2], "mean_steps": ap[3], "std_steps": ap[4]},
            "PG": {"avg_A": pg[0], "avg_B": pg[1], "median_steps": pg[2], "mean_steps": pg[3], "std_steps": pg[4]},
        },
    }


def write_report(results: Dict[str, Dict[str, float]], out_dir: str = "training_outputs", filename_prefix: str = "coop_compete_report") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = Path(out_dir) / f"{filename_prefix}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return str(path)


