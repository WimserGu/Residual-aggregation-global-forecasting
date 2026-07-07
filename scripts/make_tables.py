"""Create compact manuscript tables from experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/reproduced"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/reproduced/tables"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coverage_summary = pd.read_csv(args.results_dir / "coverage_ladder" / "summary.csv")
    coverage_cmp = pd.read_csv(args.results_dir / "coverage_ladder" / "comparisons.csv")
    metadata = pd.read_json(args.results_dir / "coverage_ladder" / "metadata.json")
    coverage = (
        metadata[["K", "initial_training_coverage"]]
        .assign(Coverage_pct=lambda x: 100 * x["initial_training_coverage"])
        [["K", "Coverage_pct"]]
    )
    direct = coverage_summary[coverage_summary["model"] == "DirectK"].set_index("K")
    residual = coverage_summary[coverage_summary["model"] == "PooledResidual"].set_index("K")
    main = pd.DataFrame(
        {
            "K": direct.index,
            "Coverage_pct": coverage.set_index("K").loc[direct.index, "Coverage_pct"].values,
            "DirectK_MAE": direct["mae"].values,
            "DirectK_Residual_MAE": residual.loc[direct.index, "mae"].values,
        }
    )
    main["MAE_improvement"] = main["DirectK_MAE"] - main["DirectK_Residual_MAE"]
    main["relative_improvement_pct"] = 100 * main["MAE_improvement"] / main["DirectK_MAE"]
    main.to_csv(args.output_dir / "table_main_results.csv", index=False)

    coverage_intervals = coverage_cmp[["K", "ci_2_5", "ci_97_5", "probability_pooled_better"]].copy()
    coverage_intervals = coverage_intervals.merge(coverage, on="K", how="left")
    coverage_intervals["MAE_improvement_ci_low"] = -coverage_intervals["ci_97_5"]
    coverage_intervals["MAE_improvement_ci_high"] = -coverage_intervals["ci_2_5"]
    coverage_intervals = coverage_intervals[
        ["K", "Coverage_pct", "MAE_improvement_ci_low", "MAE_improvement_ci_high", "probability_pooled_better"]
    ]
    coverage_intervals.to_csv(args.output_dir / "table_coverage_improvement_intervals.csv", index=False)

    representation_path = args.results_dir / "representation_ablation" / "summary.csv"
    if representation_path.exists():
        pd.read_csv(representation_path).to_csv(args.output_dir / "table_representation_ablation.csv", index=False)

    placebo_path = args.results_dir / "placebos_multiseed" / "summary.csv"
    if placebo_path.exists():
        placebo = pd.read_csv(placebo_path)
        placebo["placebo"] = placebo["placebo"].replace(
            {
                "ShuffledResidual": "Shuffled residual",
                "SyntheticResidual": "Synthetic residual",
                "DuplicateAuxiliary": "Duplicated auxiliary series",
                "ObservedBrandAggregate": "Observed-brand aggregate",
            }
        )
        placebo.to_csv(args.output_dir / "table_placebo_robustness.csv", index=False)

    print(f"Wrote tables to {args.output_dir}")


if __name__ == "__main__":
    main()
