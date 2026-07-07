"""Run the Top-K coverage ladder experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from residual_aggregation.experiments import run_coverage_ladder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/monthly_market_series.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/reproduced/coverage_ladder"))
    args = parser.parse_args()
    run_coverage_ladder(args.data, args.output_dir)


if __name__ == "__main__":
    main()
