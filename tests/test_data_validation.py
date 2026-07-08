from __future__ import annotations

import pandas as pd
import pytest

from residual_aggregation.config import BRANDS
from residual_aggregation.data import load_processed, write_synthetic_sample


def test_synthetic_sample_loads_as_balanced_panel(tmp_path):
    path = tmp_path / "monthly_market_series.csv"
    write_synthetic_sample(path, months=72)

    wide, shares, monthly = load_processed(path)

    assert list(wide.columns) == BRANDS
    assert wide.shape == (72, len(BRANDS))
    assert shares.shape == wide.shape
    assert len(monthly) == 72


def test_load_processed_rejects_inconsistent_monthly_market_size(tmp_path):
    path = tmp_path / "bad_market_size.csv"
    rows = []
    for brand in BRANDS:
        rows.append(
            {
                "date": "2018-01-01",
                "brand": brand,
                "sales": 10,
                "market_size": 100 if brand != BRANDS[-1] else 101,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)

    with pytest.raises(ValueError, match="market_size must be constant"):
        load_processed(path)


def test_load_processed_rejects_duplicate_brand_month(tmp_path):
    path = tmp_path / "duplicates.csv"
    rows = []
    for brand in BRANDS:
        rows.append({"date": "2018-01-01", "brand": brand, "sales": 10, "market_size": 100})
    rows.append({"date": "2018-01-01", "brand": BRANDS[0], "sales": 10, "market_size": 100})
    pd.DataFrame(rows).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate brand-month"):
        load_processed(path)
