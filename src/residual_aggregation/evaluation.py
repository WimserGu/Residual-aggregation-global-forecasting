"""Evaluation and bootstrap utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BOOTSTRAP_REPS, DATE, RANDOM_SEED


def monthly_mae(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly MAE by K and model."""
    work = predictions.copy()
    work["ae"] = np.abs(work["actual"] - work["prediction"])
    return work.groupby(["K", "model", DATE], as_index=False)["ae"].mean()


def paired_bootstrap(
    monthly: pd.DataFrame,
    k: int,
    left: str,
    right: str,
    seed: int = RANDOM_SEED,
    reps: int = BOOTSTRAP_REPS,
) -> dict[str, float]:
    """Bootstrap paired monthly MAE differences for two models."""
    subset = monthly[(monthly["K"] == k) & monthly["model"].isin([left, right])]
    pivot = subset.pivot(index=DATE, columns="model", values="ae").dropna()
    if pivot.empty:
        raise ValueError(f"No paired monthly errors available for K={k}, {left} vs {right}.")
    if left not in pivot.columns or right not in pivot.columns:
        raise ValueError(f"Both models must be present for paired bootstrap: {left}, {right}.")

    diff = (pivot[right] - pivot[left]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed + k + sum(ord(c) for c in right))
    samples = rng.choice(diff, size=(reps, len(diff)), replace=True).mean(axis=1)
    return {
        "K": k,
        "left": left,
        "right": right,
        "right_minus_left": float(diff.mean()),
        "share_months_right_better": float(np.mean(diff < 0)),
        "ci_2_5": float(np.quantile(samples, 0.025)),
        "ci_97_5": float(np.quantile(samples, 0.975)),
        "probability_right_better": float(np.mean(samples < 0)),
    }
