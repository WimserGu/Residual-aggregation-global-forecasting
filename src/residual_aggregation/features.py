"""Feature construction for component-level global forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATE


def component_sales_features(wide: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Construct lag, rolling-mean, seasonal, and component-identity features."""
    long = (
        wide.stack()
        .rename("component_sales")
        .reset_index()
        .rename(columns={"brand": "component"})
        .sort_values(["component", DATE])
        .reset_index(drop=True)
    )
    group = long.groupby("component", sort=False)["component_sales"]
    for lag in [1, 2, 3, 6, 12]:
        long[f"component_lag_{lag}"] = group.shift(lag)
    long["component_roll3"] = group.transform(lambda z: z.shift(1).rolling(3).mean())
    long["component_roll6"] = group.transform(lambda z: z.shift(1).rolling(6).mean())
    long["sin_month"] = np.sin(2 * np.pi * long[DATE].dt.month / 12)
    long["cos_month"] = np.cos(2 * np.pi * long[DATE].dt.month / 12)

    dummies = pd.get_dummies(long["component"], prefix="component", dtype=float)
    long = pd.concat([long, dummies], axis=1)
    features = [
        "component_lag_1",
        "component_lag_2",
        "component_lag_3",
        "component_lag_6",
        "component_lag_12",
        "component_roll3",
        "component_roll6",
        "sin_month",
        "cos_month",
        *dummies.columns.tolist(),
    ]
    return long, features


def add_aggregate_predictors(frame: pd.DataFrame) -> pd.DataFrame:
    """Add lagged residual and selected-total predictors for representation ablation."""
    monthly = (
        frame.groupby(DATE, as_index=False)
        .agg(selected_sales=("sales", "sum"), market_size=("market_size", "first"))
        .sort_values(DATE)
    )
    monthly["residual_sales"] = monthly["market_size"] - monthly["selected_sales"]
    for source in ["residual_sales", "selected_sales"]:
        for lag in [1, 2, 3, 6, 12]:
            monthly[f"{source}_lag_{lag}"] = monthly[source].shift(lag)
        monthly[f"{source}_roll3"] = monthly[source].shift(1).rolling(3).mean()
        monthly[f"{source}_roll6"] = monthly[source].shift(1).rolling(6).mean()
    keep = [DATE] + [
        column
        for column in monthly.columns
        if column.startswith("residual_sales_lag_")
        or column.startswith("selected_sales_lag_")
        or column in {
            "residual_sales_roll3",
            "residual_sales_roll6",
            "selected_sales_roll3",
            "selected_sales_roll6",
        }
    ]
    return frame.merge(monthly[keep], on=DATE, how="left", validate="many_to_one")
