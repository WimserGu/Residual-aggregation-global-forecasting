from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from residual_aggregation.config import BRANDS, DATE
from residual_aggregation.experiments import selected_brands
from residual_aggregation.features import component_sales_features
from residual_aggregation.model import smape


def test_component_features_use_only_past_sales():
    dates = pd.date_range("2020-01-01", periods=15, freq="MS")
    wide = pd.DataFrame({"A": np.arange(1, 16, dtype=float)}, index=dates)
    wide.index.name = DATE
    wide.columns.name = "brand"

    features, _ = component_sales_features(wide)
    row = features[(features[DATE] == dates[12]) & (features["component"] == "A")].iloc[0]

    assert row["component_sales"] == 13.0
    assert row["component_lag_1"] == 12.0
    assert row["component_lag_12"] == 1.0
    assert row["component_roll3"] == pytest.approx(np.mean([10.0, 11.0, 12.0]))


def test_smape_returns_zero_when_actual_and_prediction_are_zero():
    actual = np.array([0.0, 0.0])
    predicted = np.array([0.0, 0.0])

    assert smape(actual, predicted) == 0.0


def test_selected_brands_uses_initial_window_only():
    dates = pd.date_range("2020-01-01", periods=4, freq="MS")
    shares = pd.DataFrame(0.01, index=dates, columns=BRANDS)
    shares.loc[dates[:2], "Audi"] = 0.4
    shares.loc[dates[:2], "BMW"] = 0.3
    shares.loc[dates[2:], "Tesla"] = 0.9

    assert selected_brands(shares, 2, list(dates[:2])) == ["Audi", "BMW"]
