"""Run placebo auxiliary-series experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from residual_aggregation.experiments import run_placebos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/monthly_market_series.csv"))
    parser.add_argument(
        "--coverage-predictions",
        type=Path,
        default=Path("results/reproduced/coverage_ladder/predictions.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/reproduced/placebos_multiseed"))
    parser.add_argument("--single-seed", action="store_true")
    parser.add_argument("--without-observed-aggregate", action="store_true")
    args = parser.parse_args()
    run_placebos(
        args.data,
        args.coverage_predictions,
        args.output_dir,
        multiseed=not args.single_seed,
        include_observed_aggregate=not args.without_observed_aggregate,
    )


if __name__ == "__main__":
    main()
