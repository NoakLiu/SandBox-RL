#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure project root on path when run directly
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import core module directly to avoid heavy package __init__
CORE_DIR = Path(PROJECT_ROOT) / "sandgraph" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from coop_compete_benchmark import run_benchmark, write_report, run_benchmark_suite, write_suite_csv  # type: ignore
import json


def main():
    # Single-setting report
    results = run_benchmark(runs=5, episodes=200, horizon=32, coop_level=0.6, difficulty=0.3, seed=42)
    out_path = write_report(results, out_dir="training_outputs", filename_prefix="coop_compete_core")
    print(f"Report written to: {out_path}")

    # Suite report across regimes (JSON + CSV)
    suite = run_benchmark_suite()
    suite_path = Path("training_outputs") / "coop_compete_core_suite.json"
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    with open(suite_path, "w", encoding="utf-8") as f:
        json.dump(suite, f, ensure_ascii=False, indent=2)
    print(f"Suite report written to: {suite_path}")

    csv_path = write_suite_csv(suite, out_path="training_outputs/coop_compete_core_suite.csv")
    print(f"Suite CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
