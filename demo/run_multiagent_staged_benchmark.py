#!/usr/bin/env python3
import sys
from pathlib import Path
import json

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CORE_DIR = Path(PROJECT_ROOT) / "sandbox_rl" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from coop_compete_benchmark import run_multiagent_benchmark  # type: ignore
import matplotlib.pyplot as plt


def main():
    # Single common initialization for fair comparison across strategies
    res = run_multiagent_benchmark(num_agents=8, episodes=100, horizon=16, difficulty=0.3, warmup_episodes=20, seed=11)
    out_dir = Path("training_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "multiagent_staged_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"JSON written: {out_json}")

    # Plot curves
    curves = res["curves"]
    plt.figure(figsize=(7, 4))
    for k, ys in curves.items():
        plt.plot(range(1, len(ys) + 1), ys, label=k)
    plt.axvline(x=res["params"]["warmup_episodes"], color="gray", linestyle="--", label="warmup end")
    plt.xlabel("Episode")
    plt.ylabel("Avg reward per step (mean across 8 agents)")
    plt.title("8-agent staged benchmark (warmup equal rewards, then diverge)")
    plt.legend()
    png = Path("visualization_outputs") / "multiagent_staged_benchmark.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png, dpi=150)
    plt.close()
    print(f"Plot written: {png}")


if __name__ == "__main__":
    main()
