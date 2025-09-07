#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure project root on path when run directly
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sandgraph.core.coop_compete_benchmark import run_benchmark, write_report


def main():
    results = run_benchmark(runs=5, episodes=200, horizon=32, coop_level=0.6, difficulty=0.3, seed=42)
    out_path = write_report(results, out_dir="training_outputs", filename_prefix="coop_compete_core")
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
