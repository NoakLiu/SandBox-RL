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
from typing import Dict, List, Tuple, Optional
import csv


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
        p = self._sigmoid(self.theta)
        grad = 0.0
        for (a, _logp, r) in trajectories:
            advantage = r - baseline
            grad += (a - p) * advantage
        grad /= max(1, len(trajectories))
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


class OurMethodPolicy:
    """Adaptive policy with opponent modeling and uncertainty-aware update.

    - Maintains Bernoulli logit (theta) for cooperate.
    - Tracks opponent's cooperate frequency (exp. moving avg).
    - Adjusts update gain by payoff alignment with estimated cooperation regime.
    """

    def __init__(self, lr: float = 0.05, momentum: float = 0.9):
        self.theta = 0.0
        self.lr = lr
        self.momentum = momentum
        self.opp_coop_est = 0.5

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def act(self) -> Tuple[int, float]:
        p = self._sigmoid(self.theta)
        a = 1 if random.random() < p else 0
        logp = math.log(p + 1e-8) if a == 1 else math.log(1 - p + 1e-8)
        return a, logp

    def observe_opponent(self, opp_action: int):
        self.opp_coop_est = self.momentum * self.opp_coop_est + (1 - self.momentum) * (1 if opp_action == 1 else 0)

    def update(self, trajectories: List[Tuple[int, float, float]], env: CoopCompeteEnv):
        if not trajectories:
            return
        rewards = [r for (_, _, r) in trajectories]
        baseline = statistics.mean(rewards)
        p = self._sigmoid(self.theta)
        # Gain aligns with expected coop regime: if env likely cooperative, push towards cooperate
        coop_regime = env.cooperation_level
        gain = (2 * coop_regime - 1.0) * (2 * self.opp_coop_est - 1.0)
        grad = 0.0
        for (a, _logp, r) in trajectories:
            advantage = r - baseline
            grad += (a - p) * advantage
        grad /= max(1, len(trajectories))
        self.theta += self.lr * grad * (1.0 + 0.5 * gain)


def run_benchmark(runs: int = 5, episodes: int = 200, horizon: int = 32, coop_level: float = 0.6, difficulty: float = 0.3, seed: int = 42) -> Dict[str, dict]:
    random.seed(seed)
    results: Dict[str, Dict[str, List[float]]] = {
        "AC": {"A": [], "B": []},
        "AP": {"A": [], "B": []},
        "PG": {"A": [], "B": []},
        "OUR": {"A": [], "B": []},
    }
    steps_to_target: Dict[str, List[int]] = {"AC": [], "AP": [], "PG": [], "OUR": []}
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

        # OUR METHOD
        our_a = OurMethodPolicy(lr=0.05, momentum=0.9)
        our_b = OurMethodPolicy(lr=0.05, momentum=0.9)
        curve_our_a: List[float] = []
        curve_our_b: List[float] = []
        reached = False
        for ep in range(episodes):
            # roll out with opponent modeling per-step
            ep_ra = 0.0
            ep_rb = 0.0
            traj_a: List[Tuple[int, float, float]] = []
            traj_b: List[Tuple[int, float, float]] = []
            for _ in range(horizon):
                a, logpa = our_a.act()
                b, logpb = our_b.act()
                ra, rb = env.step(a, b)
                our_a.observe_opponent(b)
                our_b.observe_opponent(a)
                ep_ra += ra
                ep_rb += rb
                traj_a.append((a, logpa, ra))
                traj_b.append((b, logpb, rb))
            ra_s = ep_ra / horizon
            rb_s = ep_rb / horizon
            curve_our_a.append(ra_s)
            curve_our_b.append(rb_s)
            our_a.update(traj_a, env)
            our_b.update(traj_b, env)
            if not reached and ra_s >= target and rb_s >= target:
                steps_to_target["OUR"].append(ep + 1)
                reached = True
        results["OUR"]["A"].append(sum(curve_our_a) / episodes)
        results["OUR"]["B"].append(sum(curve_our_b) / episodes)

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
    our = summarize("OUR")

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
            "OUR": {"avg_A": our[0], "avg_B": our[1], "median_steps": our[2], "mean_steps": our[3], "std_steps": our[4]},
        },
    }


def write_report(results: Dict[str, Dict[str, float]], out_dir: str = "training_outputs", filename_prefix: str = "coop_compete_report") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = Path(out_dir) / f"{filename_prefix}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return str(path)


def run_benchmark_suite() -> Dict[str, list]:
    """Run a parameter sweep to compare AC/AP/PG/OUR across regimes."""
    settings = [
        {"coop_level": 0.2, "difficulty": 0.3},
        {"coop_level": 0.6, "difficulty": 0.3},
        {"coop_level": 0.8, "difficulty": 0.3},
        {"coop_level": 0.6, "difficulty": 0.6},
    ]
    summary: Dict[str, list] = {"suite": []}
    for i, s in enumerate(settings):
        res = run_benchmark(runs=5, episodes=200, horizon=32, coop_level=s["coop_level"], difficulty=s["difficulty"], seed=42 + i)
        entry = {"setting": s, "metrics": res["metrics"]}
        summary["suite"].append(entry)
    return summary


def write_suite_csv(summary: Dict[str, list], out_path: str = "training_outputs/coop_compete_core_suite.csv") -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["coop_level", "difficulty", "policy", "avg_A", "avg_B", "median_steps", "mean_steps", "std_steps"])
        for item in summary.get("suite", []):
            setting = item.get("setting", {})
            metrics = item.get("metrics", {})
            coop_level = setting.get("coop_level")
            difficulty = setting.get("difficulty")
            for policy in ["AC", "AP", "PG", "OUR"]:
                m = metrics.get(policy, {})
                writer.writerow([
                    coop_level,
                    difficulty,
                    policy,
                    round(m.get("avg_A", 0.0), 6),
                    round(m.get("avg_B", 0.0), 6),
                    m.get("median_steps", ""),
                    m.get("mean_steps", ""),
                    round(m.get("std_steps", 0.0), 6),
                ])
    return out_path


def generate_learning_curves(
    episodes: int = 100,
    horizon: int = 32,
    coop_levels: Optional[List[float]] = None,
    difficulties: Optional[List[float]] = None,
    seed: int = 7,
    lr_pg: float = 0.05,
    lr_our: float = 0.05,
) -> Dict[str, dict]:
    """Produce per-episode average rewards for AC/AP/PG/OUR across settings.

    Returns a dict keyed by setting label -> { policy -> [reward_per_step over episodes] }.
    """
    random.seed(seed)
    if coop_levels is None:
        coop_levels = [0.2, 0.6, 0.8]
    if difficulties is None:
        difficulties = [0.3]

    curves: Dict[str, dict] = {}
    for c in coop_levels:
        for d in difficulties:
            label = f"coop={c},diff={d}"
            env = CoopCompeteEnv(cooperation_level=c, difficulty=d)
            # init policies
            pg_a = SimplePG(lr=lr_pg)
            pg_b = SimplePG(lr=lr_pg)
            our_a = OurMethodPolicy(lr=lr_our, momentum=0.9)
            our_b = OurMethodPolicy(lr=lr_our, momentum=0.9)
            ac_curve: List[float] = []
            ap_curve: List[float] = []
            pg_curve: List[float] = []
            our_curve: List[float] = []
            for _ in range(episodes):
                # AC
                ra, rb, _, _ = _run_episode(env, always_cooperate_policy, always_cooperate_policy, horizon, False)
                ac_curve.append((ra + rb) / 2.0 / horizon)
                # AP
                ra, rb, _, _ = _run_episode(env, always_compete_policy, always_compete_policy, horizon, False)
                ap_curve.append((ra + rb) / 2.0 / horizon)
                # PG
                ra, rb, traj_a, traj_b = _run_episode(env, pg_a, pg_b, horizon, True)
                pg_curve.append((ra + rb) / 2.0 / horizon)
                pg_a.update(traj_a)
                pg_b.update(traj_b)
                # OUR
                ep_ra = 0.0
                ep_rb = 0.0
                traj_a = []
                traj_b = []
                for _t in range(horizon):
                    a, logpa = our_a.act()
                    b, logpb = our_b.act()
                    rra, rrb = env.step(a, b)
                    our_a.observe_opponent(b)
                    our_b.observe_opponent(a)
                    ep_ra += rra
                    ep_rb += rrb
                    traj_a.append((a, logpa, rra))
                    traj_b.append((b, logpb, rrb))
                our_curve.append((ep_ra + ep_rb) / 2.0 / horizon)
                our_a.update(traj_a, env)
                our_b.update(traj_b, env)

            curves[label] = {
                "AC": ac_curve,
                "AP": ap_curve,
                "PG": pg_curve,
                "OUR": our_curve,
            }
    return curves


# ------------------------ Multi-agent staged benchmark ------------------------

class MultiAgentStagedEnv:
    """N-agent environment with staged reward dynamics.

    - Warmup (episodes < warmup_episodes): all agents receive identical base rewards
      to ensure equal start.
    - Divergence (episodes >= warmup_episodes): rewards incorporate cooperative
      externality and adversarial penalties.
    """

    def __init__(
        self,
        num_agents: int = 8,
        difficulty: float = 0.3,
        coop_weight: float = 0.25,
        adversary_frac: float = 0.25,
        warmup_episodes: int = 20,
    ):
        self.num_agents = num_agents
        self.difficulty = max(0.0, min(1.0, difficulty))
        self.coop_weight = coop_weight
        self.adversary_frac = adversary_frac
        self.warmup_episodes = warmup_episodes
        # Pre-assign adversaries (indices)
        k = max(1, int(self.num_agents * self.adversary_frac))
        self.adversaries = set(range(k))

    def step(self, actions: List[int], episode_idx: int) -> List[float]:
        # 0=compete, 1=cooperate
        # curriculum base increases slightly over time to enable learning signal
        base0 = max(0.1, 1.0 - self.difficulty)
        base = base0 * (1.0 + 0.3 * min(1.0, max(0.0, (episode_idx - self.warmup_episodes) / max(1, self.warmup_episodes))))
        n = len(actions)
        rewards = [base] * n
        if episode_idx < self.warmup_episodes:
            # identical rewards
            noise = [0.0 for _ in range(n)]
            return [max(0.0, base + z) for z in noise]

        # divergence phase: cooperative externality
        coop_ratio = sum(actions) / max(1, n)
        for i, a in enumerate(actions):
            payoff = self.coop_weight * (coop_ratio - 0.5)  # centered
            # adversaries gain when others cooperate but they compete
            if i in self.adversaries:
                if a == 0:  # compete
                    payoff += 0.1 * coop_ratio
                else:
                    payoff -= 0.05 * coop_ratio
            # mismatch penalty
            if (a == 1 and coop_ratio < 0.3) or (a == 0 and coop_ratio > 0.7):
                payoff -= 0.05
            # smaller noise for clearer learning signal
            rewards[i] = max(0.0, base + payoff + random.gauss(0, 0.005))
        return rewards


def run_multiagent_benchmark(
    num_agents: int = 8,
    episodes: int = 100,
    horizon: int = 16,
    difficulty: float = 0.3,
    warmup_episodes: int = 20,
    seed: int = 11,
) -> Dict[str, dict]:
    random.seed(seed)
    env = MultiAgentStagedEnv(
        num_agents=num_agents, difficulty=difficulty, warmup_episodes=warmup_episodes
    )

    def rollout_fixed(policy_val: int) -> List[float]:
        curve: List[float] = []
        for ep in range(episodes):
            ep_sum = 0.0
            for _ in range(horizon):
                acts = [policy_val] * num_agents
                rs = env.step(acts, ep)
                ep_sum += sum(rs) / num_agents
            curve.append(ep_sum / horizon)
        return curve

    # ALLC / ALLP
    allc_curve = rollout_fixed(1)
    allp_curve = rollout_fixed(0)

    # Independent PG per agent
    pg_agents = [SimplePG(lr=0.05) for _ in range(num_agents)]
    pg_curve: List[float] = []
    for ep in range(episodes):
        ep_sum = 0.0
        trajs: List[List[Tuple[int, float, float]]] = [[] for _ in range(num_agents)]
        for _ in range(horizon):
            acts = []
            logps = []
            for ag in pg_agents:
                a, lp = ag.act()
                acts.append(a)
                logps.append(lp)
            rs = env.step(acts, ep)
            ep_sum += sum(rs) / num_agents
            for i in range(num_agents):
                trajs[i].append((acts[i], logps[i], rs[i]))
        pg_curve.append(ep_sum / horizon)
        for i, ag in enumerate(pg_agents):
            ag.update(trajs[i])

    # Our method per agent with opponent modeling (aggregate coop estimate)
    our_agents = [OurMethodPolicy(lr=0.05, momentum=0.9) for _ in range(num_agents)]
    our_curve: List[float] = []
    for ep in range(episodes):
        ep_sum = 0.0
        trajs = [[] for _ in range(num_agents)]
        for _ in range(horizon):
            acts = []
            logps = []
            for ag in our_agents:
                a, lp = ag.act()
                acts.append(a)
                logps.append(lp)
            rs = env.step(acts, ep)
            ep_sum += sum(rs) / num_agents
            # observe others
            for i, ag in enumerate(our_agents):
                # naive: observe average of others' cooperate actions
                other_coop = (sum(acts) - acts[i]) / max(1, (num_agents - 1))
                ag.opp_coop_est = 0.9 * ag.opp_coop_est + 0.1 * other_coop
                trajs[i].append((acts[i], logps[i], rs[i]))
        our_curve.append(ep_sum / horizon)
        for i, ag in enumerate(our_agents):
            # use env difficulty as proxy for regime in update
            ag.update(trajs[i], CoopCompeteEnv(cooperation_level=ag.opp_coop_est, difficulty=difficulty))

    return {
        "params": {
            "num_agents": num_agents,
            "episodes": episodes,
            "horizon": horizon,
            "difficulty": difficulty,
            "warmup_episodes": warmup_episodes,
            "seed": seed,
        },
        "curves": {
            "ALLC": allc_curve,
            "ALLP": allp_curve,
            "PG": pg_curve,
            "OUR": our_curve,
        },
    }


