"""Forecasting model helpers."""

from __future__ import annotations

import math

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import DATE, LGB_PARAMS


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric mean absolute percentage error."""
    denom = np.abs(actual) + np.abs(predicted)
    valid = denom > 0
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
    months = sorted(train[DATE].unique())
    validation = set(months[-12:])
    fit = train[~train[DATE].isin(validation)]
    valid = train[train[DATE].isin(validation)]
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
