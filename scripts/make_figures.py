"""Generate manuscript figures from result CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 11,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def figure_accuracy(results_dir: Path, output_dir: Path) -> None:
    summary = pd.read_csv(results_dir / "coverage_ladder" / "summary.csv")
    metadata = pd.read_json(results_dir / "coverage_ladder" / "metadata.json")
    coverage = metadata.set_index("K")["initial_training_coverage"].mul(100)
    direct = summary[summary["model"] == "DirectK"].set_index("K")
    residual = summary[summary["model"] == "PooledResidual"].set_index("K")
    k_values = direct.index.to_numpy()
    x = coverage.loc[k_values].to_numpy()
    y_direct = direct.loc[k_values, "mae"].to_numpy()
    y_residual = residual.loc[k_values, "mae"].to_numpy()
    improvement_pct = 100 * (y_direct - y_residual) / y_direct

    fig, ax = plt.subplots(figsize=(7.1, 4.35))
    ax.plot(x, y_direct, color="#111111", linestyle="-", marker="o", markerfacecolor="white", label="DirectK")
    ax.plot(
        x,
        y_residual,
        color="#555555",
        linestyle=(0, (4, 2.4)),
        marker="s",
        markerfacecolor="#d9d9d9",
        label="DirectK + Residual",
    )
    for x_i, y_i, k in zip(x, np.maximum(y_direct, y_residual), k_values):
        ax.text(x_i, y_i + 14, f"K={k}", ha="center", va="bottom", fontsize=8.2)
    ax.set_title("Forecast accuracy across market coverage levels", pad=9, fontweight="bold")
    ax.set_xlabel("Observed market coverage (%)")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.00), frameon=True)

    inset = ax.inset_axes([0.62, 0.66, 0.28, 0.23])
    inset.plot(x, improvement_pct, color="#222222", marker="^", markerfacecolor="white", linewidth=0.95)
    inset.set_title("Improvement (%)", fontsize=7.6, pad=3)
    inset.set_xticks([round(x[0]), round(x[1]), round(x[-1])])
    inset.set_yticks([0, 2, 4])
    inset.grid(axis="y", color="#e0e0e0", linewidth=0.40)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0.00, 0.00, 0.84, 1.00])
    fig.savefig(output_dir / "figure2b_accuracy_levels.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "figure2b_accuracy_levels.svg", bbox_inches="tight")
    plt.close(fig)


def figure_coverage_improvement(results_dir: Path, output_dir: Path) -> None:
    comparisons = pd.read_csv(results_dir / "coverage_ladder" / "comparisons.csv")
    metadata = pd.read_json(results_dir / "coverage_ladder" / "metadata.json")
    coverage = metadata.set_index("K")["initial_training_coverage"].mul(100)
    comparisons["improvement"] = -comparisons["right_minus_left"]
    comparisons["ci_low"] = -comparisons["ci_97_5"]
    comparisons["ci_high"] = -comparisons["ci_2_5"]
    x = coverage.loc[comparisons["K"]].to_numpy()
    y = comparisons["improvement"].to_numpy()
    yerr = np.vstack([y - comparisons["ci_low"].to_numpy(), comparisons["ci_high"].to_numpy() - y])

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        color="#111111",
        marker="o",
        markerfacecolor="white",
        capsize=3,
        linewidth=1.1,
    )
    for x_i, y_i, k in zip(x, y, comparisons["K"]):
        ax.text(x_i, y_i + 3, f"K={k}", ha="center", va="bottom", fontsize=8.2)
    ax.set_title("Residual aggregation performance across market coverage levels", pad=9, fontweight="bold")
    ax.set_xlabel("Observed market coverage (%)")
    ax.set_ylabel("MAE improvement")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "figure2_coverage_performance.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "figure2_coverage_performance.svg", bbox_inches="tight")
    plt.close(fig)


def figure_placebos(results_dir: Path, output_dir: Path) -> None:
    path = results_dir / "placebos_multiseed" / "seed_results.csv"
    if not path.exists():
        return
    seeds = pd.read_csv(path)
    seeds = seeds[seeds["placebo"].isin(["ShuffledResidual", "SyntheticResidual"])]
    seeds["advantage"] = seeds["placebo_minus_real_mae"]
    labels = []
    data = []
    positions = []
    pos = 1
    for k in sorted(seeds["K"].unique()):
        for placebo in ["ShuffledResidual", "SyntheticResidual"]:
            subset = seeds[(seeds["K"] == k) & (seeds["placebo"] == placebo)]
            data.append(subset["advantage"].to_numpy())
            labels.append("Against shuffled residual" if placebo == "ShuffledResidual" else "Against synthetic residual")
            positions.append(pos)
            pos += 1
        pos += 0.6

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0, color="#777777", linewidth=0.8)
    box = ax.boxplot(data, positions=positions, widths=0.52, patch_artist=True, showfliers=False)
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor("#f2f2f2" if index % 2 == 0 else "#d9d9d9")
        patch.set_edgecolor("#333333")
    for element in ["whiskers", "caps", "medians"]:
        for item in box[element]:
            item.set_color("#333333")
            item.set_linewidth(0.8)
    centers = [(positions[i] + positions[i + 1]) / 2 for i in range(0, len(positions), 2)]
    ax.set_xticks(centers)
    ax.set_xticklabels([f"K={k}" for k in sorted(seeds["K"].unique())])
    ax.set_ylabel("MAE advantage of authentic residual")
    ax.set_title("Mechanism validation: information completion versus auxiliary-task regularization", pad=9, fontweight="bold")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        [box["boxes"][0], box["boxes"][1]],
        ["Against shuffled residual", "Against synthetic residual"],
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "figure3_mechanism_placebos.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "figure3_mechanism_placebos.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/reproduced"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/reproduced"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    figure_accuracy(args.results_dir, args.output_dir)
    figure_coverage_improvement(args.results_dir, args.output_dir)
    figure_placebos(args.results_dir, args.output_dir)
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
