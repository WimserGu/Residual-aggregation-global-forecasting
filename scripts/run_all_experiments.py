"""Run the full reproduction workflow from processed data.

The script assumes that ``data/processed/monthly_market_series.csv`` exists.
Use ``scripts/prepare_processed_data.py`` to create it from licensed raw data,
or to create a synthetic sample for a smoke test.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    """Run one Python command and stop if it fails."""
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/monthly_market_series.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/reproduced"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/reproduced"))
    parser.add_argument("--skip-placebos", action="store_true", help="Skip the slower 20-seed placebo experiments.")
    args = parser.parse_args()

    python = sys.executable
    if not args.data.exists():
        raise FileNotFoundError(
            f"{args.data} not found. Prepare processed data before running the full workflow."
        )

    coverage_dir = args.results_dir / "coverage_ladder"
    ablation_dir = args.results_dir / "representation_ablation"
    placebo_dir = args.results_dir / "placebos_multiseed"
    tables_dir = args.results_dir / "tables"

    run(
        [
            python,
            str(REPO_ROOT / "scripts" / "run_coverage_ladder.py"),
            "--data",
            str(args.data),
            "--output-dir",
            str(coverage_dir),
        ]
    )
    run(
        [
            python,
            str(REPO_ROOT / "scripts" / "run_representation_ablation.py"),
            "--data",
            str(args.data),
            "--output-dir",
            str(ablation_dir),
        ]
    )
    if not args.skip_placebos:
        run(
            [
                python,
                str(REPO_ROOT / "scripts" / "run_placebos.py"),
                "--data",
                str(args.data),
                "--coverage-predictions",
                str(coverage_dir / "predictions.csv"),
                "--output-dir",
                str(placebo_dir),
            ]
        )
    run(
        [
            python,
            str(REPO_ROOT / "scripts" / "make_tables.py"),
            "--results-dir",
            str(args.results_dir),
            "--output-dir",
            str(tables_dir),
        ]
    )
    run(
        [
            python,
            str(REPO_ROOT / "scripts" / "make_figures.py"),
            "--results-dir",
            str(args.results_dir),
            "--output-dir",
            str(args.figures_dir),
        ]
    )


if __name__ == "__main__":
    main()
