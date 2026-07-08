"""Experiment definitions for residual aggregation under partial market coverage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    BRANDS,
    DATE,
    K_VALUES,
    MARKET_SIZE,
    N_PLACEBO_REPLICATES,
    RANDOM_SEED,
    TEST_START_INDEX,
)
from .data import load_processed
from .evaluation import monthly_mae, paired_bootstrap
from .features import add_aggregate_predictors, component_sales_features
from .model import lgb_predict, metric_row


def selected_brands(shares: pd.DataFrame, k: int, initial_months: list[pd.Timestamp]) -> list[str]:
    """Select the K largest brands by average share in the initial training window."""
    if k < 1 or k > len(BRANDS):
        raise ValueError(f"k must be between 1 and {len(BRANDS)}; got {k}.")
    if not initial_months:
        raise ValueError("initial_months must not be empty.")
    missing_months = [month for month in initial_months if month not in shares.index]
    if missing_months:
        raise ValueError(f"Initial training months are missing from shares index: {missing_months[:3]}")

    ranking = shares.loc[initial_months, BRANDS].mean().sort_values(ascending=False)
    return ranking.head(k).index.tolist()


def fit_predict_components(
    wide: pd.DataFrame,
    ordered_components: list[str],
    test_months: list[pd.Timestamp],
) -> pd.DataFrame:
    """Run one-step rolling-origin forecasts for a component panel."""
    missing_components = [component for component in ordered_components if component not in wide.columns]
    if missing_components:
        raise ValueError(f"Component panel is missing requested components: {missing_components}")

    feature_df, features = component_sales_features(wide[ordered_components])
    rows = []
    for test_month in test_months:
        train = feature_df[feature_df[DATE] < test_month].dropna(
            subset=features + ["component_sales"]
        )
        test = feature_df[feature_df[DATE] == test_month].dropna(subset=features)
        missing_test_components = sorted(set(ordered_components).difference(test["component"]))
        if missing_test_components:
            raise ValueError(
                f"Test month {test_month.date()} is missing feature-complete rows for: {missing_test_components}"
            )
        test = test.set_index("component").loc[ordered_components].reset_index()
        prediction = np.maximum(lgb_predict(train, test, features, "component_sales"), 0.0)
        rows.append(
            pd.DataFrame(
                {
                    DATE: test_month,
                    "component": ordered_components,
                    "actual": test["component_sales"].to_numpy(dtype=float),
                    "prediction": prediction,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def run_coverage_ladder(data_path: Path, output_dir: Path) -> None:
    """Run DirectK and DirectK+Residual models across the Top-K coverage ladder."""
    full_wide, shares, monthly = load_processed(data_path)
    months = sorted(pd.Timestamp(v) for v in full_wide.index)
    if len(months) <= TEST_START_INDEX:
        raise ValueError(
            f"At least {TEST_START_INDEX + 1} months are required; found {len(months)}."
        )
    initial_months = months[:TEST_START_INDEX]
    test_months = months[TEST_START_INDEX:]
    market = monthly.set_index(DATE)[MARKET_SIZE].sort_index()
    all_rows = []
    metadata = []

    for k in K_VALUES:
        selected = selected_brands(shares, k, initial_months)
        direct_wide = full_wide[selected].copy()
        pooled_wide = direct_wide.copy()
        pooled_wide["Residual"] = market - direct_wide.sum(axis=1)
        if (pooled_wide["Residual"] < 0).any():
            raise ValueError(f"Residual is negative for K={k}.")

        direct_predictions = fit_predict_components(direct_wide, selected, test_months)
        direct_predictions["model"] = "DirectK"
        pooled_predictions = fit_predict_components(pooled_wide, selected + ["Residual"], test_months)
        pooled_predictions = pooled_predictions[pooled_predictions["component"].isin(selected)].copy()
        pooled_predictions["model"] = "PooledResidual"
        combined = pd.concat([direct_predictions, pooled_predictions], ignore_index=True)
        combined["K"] = k
        all_rows.append(combined)
        metadata.append(
            {
                "K": k,
                "selected_brands": selected,
                "initial_training_coverage": float(shares.loc[initial_months, selected].sum(axis=1).mean()),
                "full_sample_coverage_descriptive": float(shares.loc[:, selected].sum(axis=1).mean()),
            }
        )

    predictions = pd.concat(all_rows, ignore_index=True)
    summary = summarize_predictions(predictions, ["K", "model"])
    monthly = monthly_mae(predictions)
    comparisons = []
    for k in K_VALUES:
        comparisons.append(paired_bootstrap(monthly, k, "DirectK", "PooledResidual"))

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output_dir / "comparisons.csv", index=False)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def summarize_predictions(predictions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Summarize forecast accuracy by the requested grouping columns."""
    rows = []
    for keys, group in predictions.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(metric_row(group["actual"].to_numpy(dtype=float), group["prediction"].to_numpy(dtype=float)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def synthetic_ar1(values: np.ndarray, seed: int) -> np.ndarray:
    """Generate a synthetic residual series matching basic AR(1)-like properties."""
    values = values.astype(float)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if len(values) > 2 and std > 0:
        phi = float(np.corrcoef(values[:-1], values[1:])[0, 1])
        if not np.isfinite(phi):
            phi = 0.0
    else:
        phi = 0.0
    phi = float(np.clip(phi, -0.9, 0.9))
    innovation_std = max(std * np.sqrt(max(1 - phi**2, 0.05)), 1.0)
    rng = np.random.default_rng(seed)
    result = np.empty(len(values), dtype=float)
    result[0] = max(values[0], 1.0)
    for index in range(1, len(values)):
        result[index] = mean + phi * (result[index - 1] - mean) + rng.normal(0, innovation_std)
        result[index] = max(result[index], 1.0)
    return result


def placebo_series(
    kind: str,
    residual_train: pd.Series,
    direct_wide: pd.DataFrame,
    train_months: list[pd.Timestamp],
    selected: list[str],
    seed: int,
) -> pd.Series:
    """Construct one auxiliary placebo series for a rolling-origin fold."""
    if residual_train.empty:
        raise ValueError("residual_train must not be empty.")
    if len(train_months) != len(residual_train):
        raise ValueError("train_months and residual_train must have the same length.")
    if not selected:
        raise ValueError("selected must contain at least one brand.")

    values = residual_train.to_numpy(dtype=float)
    if kind == "ShuffledResidual":
        rng = np.random.default_rng(seed)
        generated = rng.permutation(values)
    elif kind == "SyntheticResidual":
        generated = synthetic_ar1(values, seed)
    elif kind == "DuplicateAuxiliary":
        training_means = direct_wide.loc[train_months, selected].mean().sort_values()
        duplicate_brand = training_means.index[len(training_means) // 2]
        generated = direct_wide.loc[train_months, duplicate_brand].to_numpy(dtype=float)
    elif kind == "ObservedBrandAggregate":
        rng = np.random.default_rng(seed)
        subset_size = int(rng.integers(2, len(selected) + 1))
        subset = rng.choice(selected, size=subset_size, replace=False).tolist()
        aggregate = direct_wide.loc[train_months, subset].sum(axis=1).astype(float)
        if float(aggregate.mean()) > 0:
            aggregate = aggregate * (float(residual_train.mean()) / float(aggregate.mean()))
        generated = aggregate.to_numpy(dtype=float)
    else:
        raise ValueError(kind)
    return pd.Series(generated, index=train_months, name="Auxiliary")


def predict_auxiliary(
    direct_wide: pd.DataFrame,
    residual: pd.Series,
    selected: list[str],
    test_months: list[pd.Timestamp],
    kind: str,
    k: int,
    replicate: int = 0,
) -> pd.DataFrame:
    """Forecast selected brands after adding one fold-specific auxiliary series."""
    rows = []
    all_months = list(direct_wide.index)
    for fold, test_month in enumerate(test_months):
        train_months = [month for month in all_months if month < test_month]
        if not train_months:
            raise ValueError(f"No training months available before test month {test_month}.")
        generated = placebo_series(
            kind,
            residual.loc[train_months],
            direct_wide,
            train_months,
            selected,
            seed=RANDOM_SEED + 100000 * replicate + 1000 * k + fold,
        )
        augmented = direct_wide[selected].copy()
        augmented["Auxiliary"] = np.nan
        augmented.loc[train_months, "Auxiliary"] = generated
        feature_df, features = component_sales_features(augmented)
        train = feature_df[feature_df[DATE] < test_month].dropna(
            subset=features + ["component_sales"]
        )
        test = feature_df[
            (feature_df[DATE] == test_month) & feature_df["component"].isin(selected)
        ].dropna(subset=features)
        missing_test_components = sorted(set(selected).difference(test["component"]))
        if missing_test_components:
            raise ValueError(
                f"Test month {test_month.date()} is missing feature-complete selected rows for: "
                f"{missing_test_components}"
            )
        test = test.set_index("component").loc[selected].reset_index()
        prediction = np.maximum(lgb_predict(train, test, features, "component_sales"), 0.0)
        rows.append(
            pd.DataFrame(
                {
                    "K": k,
                    "model": kind,
                    DATE: test_month,
                    "component": selected,
                    "actual": test["component_sales"].to_numpy(dtype=float),
                    "prediction": prediction,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def run_placebos(
    data_path: Path,
    coverage_predictions_path: Path,
    output_dir: Path,
    multiseed: bool = True,
    include_observed_aggregate: bool = True,
) -> None:
    """Run placebo auxiliary-series experiments against the authentic residual."""
    full_wide, shares, monthly = load_processed(data_path)
    months = sorted(pd.Timestamp(v) for v in full_wide.index)
    initial_months = months[:TEST_START_INDEX]
    test_months = months[TEST_START_INDEX:]
    market = monthly.set_index(DATE)[MARKET_SIZE].sort_index()
    ladder = pd.read_csv(coverage_predictions_path)
    ladder[DATE] = pd.to_datetime(ladder[DATE])
    real = ladder[ladder["model"] == "PooledResidual"].copy()
    direct = ladder[ladder["model"] == "DirectK"].copy()

    placebos = ["ShuffledResidual", "SyntheticResidual", "DuplicateAuxiliary"]
    if include_observed_aggregate:
        placebos.append("ObservedBrandAggregate")
    replicates = range(N_PLACEBO_REPLICATES if multiseed else 1)
    records = []
    monthly_records = []
    prediction_frames = []
    for replicate in replicates:
        for k in K_VALUES:
            selected = selected_brands(shares, k, initial_months)
            direct_wide = full_wide[selected]
            residual = market - direct_wide.sum(axis=1)
            real_monthly = _monthly_mae_series(real[real["K"] == k])
            direct_monthly = _monthly_mae_series(direct[direct["K"] == k])
            for kind in placebos:
                placebo = predict_auxiliary(direct_wide, residual, selected, test_months, kind, k, replicate)
                placebo["replicate"] = replicate
                prediction_frames.append(placebo)
                placebo_monthly = _monthly_mae_series(placebo)
                aligned = pd.concat(
                    [
                        real_monthly.rename("real"),
                        placebo_monthly.rename("placebo"),
                        direct_monthly.rename("direct"),
                    ],
                    axis=1,
                ).dropna()
                diff_vs_real = aligned["placebo"] - aligned["real"]
                diff_vs_direct = aligned["placebo"] - aligned["direct"]
                records.append(
                    {
                        "replicate": replicate,
                        "K": k,
                        "placebo": kind,
                        "placebo_minus_real_mae": float(diff_vs_real.mean()),
                        "placebo_minus_direct_mae": float(diff_vs_direct.mean()),
                        "share_months_real_better": float(np.mean(diff_vs_real > 0)),
                        "share_months_placebo_beats_direct": float(np.mean(diff_vs_direct < 0)),
                    }
                )
                for month, row in aligned.iterrows():
                    monthly_records.append(
                        {
                            "replicate": replicate,
                            "K": k,
                            "placebo": kind,
                            DATE: month,
                            "real_mae": row["real"],
                            "placebo_mae": row["placebo"],
                            "direct_mae": row["direct"],
                        }
                    )

    results = pd.DataFrame(records)
    summary = (
        results.groupby(["K", "placebo"])
        .agg(
            mean_placebo_minus_real=("placebo_minus_real_mae", "mean"),
            median_placebo_minus_real=("placebo_minus_real_mae", "median"),
            q05_placebo_minus_real=("placebo_minus_real_mae", lambda x: x.quantile(0.05)),
            q95_placebo_minus_real=("placebo_minus_real_mae", lambda x: x.quantile(0.95)),
            proportion_seeds_real_better=("placebo_minus_real_mae", lambda x: float(np.mean(x > 0))),
            mean_share_months_real_better=("share_months_real_better", "mean"),
            mean_placebo_minus_direct=("placebo_minus_direct_mae", "mean"),
        )
        .reset_index()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(monthly_records).to_csv(output_dir / "monthly_results.csv", index=False)
    results.to_csv(output_dir / "seed_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)


def _monthly_mae_series(frame: pd.DataFrame) -> pd.Series:
    work = frame.copy()
    work["ae"] = np.abs(work["actual"] - work["prediction"])
    return work.groupby(DATE)["ae"].mean()


def run_representation_ablation(data_path: Path, output_dir: Path) -> None:
    """Compare auxiliary-series representation with predictor representations for K=14."""
    full_wide, shares, monthly = load_processed(data_path)
    months = sorted(pd.Timestamp(v) for v in full_wide.index)
    initial_months = months[:TEST_START_INDEX]
    test_months = months[TEST_START_INDEX:]
    market = monthly.set_index(DATE)[MARKET_SIZE].sort_index()
    selected = selected_brands(shares, 14, initial_months)

    direct_wide = full_wide[selected].copy()
    residual_wide = direct_wide.copy()
    residual_wide["Residual"] = market - direct_wide.sum(axis=1)
    direct = fit_predict_components(direct_wide, selected, test_months)
    direct["model"] = "Direct14"
    pooled = fit_predict_components(residual_wide, selected + ["Residual"], test_months)
    pooled = pooled[pooled["component"].isin(selected)].copy()
    pooled["model"] = "Pooled15"

    long = (
        direct_wide.stack()
        .rename("sales")
        .reset_index()
        .rename(columns={"brand": "component"})
        .merge(monthly, on=DATE, how="left", validate="many_to_one")
    )
    long = add_aggregate_predictors(long.rename(columns={"component": "brand"})).rename(columns={"brand": "component"})
    feature_df, base_features = component_sales_features(direct_wide)
    feature_df = feature_df.merge(
        long.drop(columns=["sales", MARKET_SIZE]), on=[DATE, "component"], how="left", validate="one_to_one"
    )
    residual_cols = [c for c in feature_df.columns if c.startswith("residual_sales_")]
    selected_cols = [c for c in feature_df.columns if c.startswith("selected_sales_")]
    specs = {
        "Direct14_ResidualPredictors": base_features + residual_cols,
        "Direct14_SelectedTotalPredictors": base_features + selected_cols,
        "Direct14_ResidualAndSelectedPredictors": base_features + residual_cols + selected_cols,
    }
    rows = []
    for test_month in test_months:
        test_base = feature_df[feature_df[DATE] == test_month]
        for model_name, features in specs.items():
            train = feature_df[feature_df[DATE] < test_month].dropna(subset=features + ["component_sales"])
            test = test_base.dropna(subset=features).set_index("component").loc[selected].reset_index()
            prediction = np.maximum(lgb_predict(train, test, features, "component_sales"), 0.0)
            rows.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        DATE: test_month,
                        "component": selected,
                        "actual": test["component_sales"].to_numpy(dtype=float),
                        "prediction": prediction,
                    }
                )
            )
    all_predictions = pd.concat([direct, pooled, *rows], ignore_index=True)
    summary = summarize_predictions(all_predictions, ["model"]).sort_values("mae")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions.to_csv(output_dir / "predictions.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
