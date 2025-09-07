#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure project root
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import core directly
CORE_DIR = Path(PROJECT_ROOT) / "sandgraph" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from coop_compete_benchmark import generate_learning_curves  # type: ignore
import matplotlib.pyplot as plt


def plot_curves(curves: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, series in curves.items():
        plt.figure(figsize=(7, 4))
        for policy, ys in series.items():
            plt.plot(range(1, len(ys) + 1), ys, label=policy)
        plt.xlabel("Episode")
        plt.ylabel("Avg reward per step (mean of A/B)")
        plt.title(f"Learning curves: {label}")
        plt.legend()
        fname = out_dir / f"learning_curves_{label.replace(',', '_').replace('=', '-')}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved: {fname}")


def main():
    curves = generate_learning_curves(episodes=100, horizon=32, coop_levels=[0.2, 0.6, 0.8], difficulties=[0.3, 0.6])
    out_dir = Path("visualization_outputs") / "coop_compete_curves"
    plot_curves(curves, out_dir)


if __name__ == "__main__":
    main()
