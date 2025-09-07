#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure core directory on path to avoid heavy package __init__ imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "sandgraph" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from coop_compete_benchmark import run_benchmark  # type: ignore


def test_convergence_and_reward_ordering():
    # High cooperation regime should favor AC and learned PG over AP on reward
    res_high = run_benchmark(runs=3, episodes=150, horizon=32, coop_level=0.8, difficulty=0.3, seed=123)
    ac = res_high["metrics"]["AC"]
    ap = res_high["metrics"]["AP"]
    pg = res_high["metrics"]["PG"]

    # Reward ordering (robust): PG should beat AP; cooperative strategies (AC or PG) should beat AP
    tol = 2e-2
    assert pg["avg_A"] + tol >= ap["avg_A"] - tol
    assert pg["avg_B"] + tol >= ap["avg_B"] - tol
    assert max(ac["avg_A"], pg["avg_A"]) + tol >= ap["avg_A"] - tol
    assert max(ac["avg_B"], pg["avg_B"]) + tol >= ap["avg_B"] - tol

    # Convergence (steps to target) should be finite for AC and PG
    assert ac["mean_steps"] < float("inf")
    assert pg["mean_steps"] < float("inf")

    # Low cooperation regime should favor AP over AC
    res_low = run_benchmark(runs=3, episodes=150, horizon=32, coop_level=0.2, difficulty=0.3, seed=321)
    ac2 = res_low["metrics"]["AC"]
    ap2 = res_low["metrics"]["AP"]
    pg2 = res_low["metrics"]["PG"]

    assert ap2["avg_A"] + tol >= ac2["avg_A"] - tol
    assert ap2["avg_B"] + tol >= ac2["avg_B"] - tol

    # PG should adapt and not be worst
    assert pg2["avg_A"] + tol >= min(ac2["avg_A"], ap2["avg_A"]) - tol
    assert pg2["avg_B"] + tol >= min(ac2["avg_B"], ap2["avg_B"]) - tol


