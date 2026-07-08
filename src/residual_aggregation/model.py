"""Forecasting model helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import DATE, LGB_PARAMS


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric mean absolute percentage error."""
    denom = np.abs(actual) + np.abs(predicted)
    valid = denom > 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(2 * np.abs(actual[valid] - predicted[valid]) / denom[valid]))


def metric_row(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, and sMAPE."""
    error = actual - predicted
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(np.mean(error**2))),
        "smape": smape(actual, predicted),
    }


def lgb_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
    """Fit the fixed LightGBM specification and predict the supplied test rows."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "LightGBM is required for model fitting. Install project dependencies with "
            "`pip install -r requirements.txt` or `pip install -e .`."
        ) from exc

    missing_train = [column for column in [*features, target, DATE] if column not in train.columns]
    missing_test = [column for column in [*features, DATE] if column not in test.columns]
    if missing_train:
        raise ValueError(f"Training data are missing required columns: {missing_train}")
    if missing_test:
        raise ValueError(f"Test data are missing required columns: {missing_test}")
    if train.empty:
        raise ValueError("Training data are empty after feature construction.")
    if test.empty:
        raise ValueError("Test data are empty after feature construction.")

    months = sorted(train[DATE].unique())
    if len(months) < 13:
        raise ValueError("At least 13 training months are required for the 12-month validation split.")
    validation = set(months[-12:])
    fit = train[~train[DATE].isin(validation)]
    valid = train[train[DATE].isin(validation)]
    if fit.empty or valid.empty:
        raise ValueError("Both fitting and validation partitions must contain rows.")

    dtrain = lgb.Dataset(fit[features], label=fit[target], feature_name=features)
    dvalid = lgb.Dataset(valid[features], label=valid[target], reference=dtrain)
    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=1200,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    return model.predict(test[features], num_iteration=model.best_iteration)
