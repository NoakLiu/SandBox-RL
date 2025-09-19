#!/usr/bin/env python3
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CORE_DIR = Path(PROJECT_ROOT) / "sandbox_rl" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from coop_compete_benchmark import run_team_battle  # type: ignore


def main():
    res = run_team_battle(episodes=100, horizon=16, seed=17)
    out_dir = Path("training_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "team_battle_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"JSON written: {out_json}")

    # Per-agent reward plots
    viz_dir = Path("visualization_outputs") / "team_battle"
    viz_dir.mkdir(parents=True, exist_ok=True)

    def plot_hist(hist: list, title: str, fname: Path):
        plt.figure(figsize=(8, 4))
        for i, series in enumerate(hist):
            plt.plot(range(1, len(series) + 1), series, label=f"agent_{i}")
        plt.xlabel("Episode")
        plt.ylabel("Reward per step")
        plt.title(title)
        plt.legend(ncol=4, fontsize=8)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved: {fname}")

    plot_hist(res["pg"], "Team battle - PG per-agent rewards", viz_dir / "pg_per_agent.png")
    plot_hist(res["our"], "Team battle - OUR per-agent rewards", viz_dir / "our_per_agent.png")


if __name__ == "__main__":
    main()
