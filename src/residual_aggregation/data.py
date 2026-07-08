"""Data loading and preparation utilities.

The public workflow uses a processed long-format CSV with one row per
brand-month and an official monthly market-size column. Raw Excel files are
optional and are not required when the processed file has already been created.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import BRANDS, BRAND, DATE, MARKET_SIZE, TARGET


def load_processed(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed data and return wide sales, shares, and monthly market data."""
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")

    frame = pd.read_csv(path)
    required = {DATE, BRAND, TARGET, MARKET_SIZE}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Processed data are missing required columns: {sorted(missing)}")

    frame[DATE] = pd.to_datetime(frame[DATE])
    frame = frame.sort_values([DATE, BRAND]).reset_index(drop=True)
    if frame.duplicated([DATE, BRAND]).any():
        raise ValueError("Processed data contain duplicate brand-month rows.")
    if frame.groupby(DATE)[MARKET_SIZE].nunique().gt(1).any():
        raise ValueError("market_size must be constant across brands within each month.")
    if (frame[[TARGET, MARKET_SIZE]] < 0).any().any():
        raise ValueError("sales and market_size must be non-negative.")

    wide = frame.pivot(index=DATE, columns=BRAND, values=TARGET).reindex(columns=BRANDS)
    if wide[BRANDS].isna().any().any():
        raise ValueError("Processed data must contain all 14 manuscript brands each month.")

    monthly = (
        frame[[DATE, MARKET_SIZE]]
        .drop_duplicates(DATE)
        .sort_values(DATE)
        .reset_index(drop=True)
    )
    market = monthly.set_index(DATE)[MARKET_SIZE]
    if (market <= 0).any():
        raise ValueError("market_size must be positive for every month.")
    shares = wide.div(market, axis=0)
    if not np.isfinite(shares.to_numpy(dtype=float)).all():
        raise ValueError("Computed market shares contain non-finite values.")
    return wide.sort_index(), shares.sort_index(), monthly


def prepare_from_excel(sales_path: Path, indicators_path: Path, output_path: Path) -> pd.DataFrame:
    """Create the processed CSV from the original Excel files, when available."""
    if not sales_path.exists():
        raise FileNotFoundError(f"Sales workbook not found: {sales_path}")
    if not indicators_path.exists():
        raise FileNotFoundError(f"Indicator workbook not found: {indicators_path}")

    sales = pd.read_excel(sales_path, sheet_name="Sheet2")
    indicators = pd.read_excel(indicators_path, sheet_name="Ark1")

    sales = sales.rename(columns={"Brand": BRAND, "Sales": TARGET, "date": DATE})
    indicators = indicators.rename(columns={"Month": DATE})
    sales_missing = {BRAND, TARGET, DATE}.difference(sales.columns)
    indicators_missing = {DATE, MARKET_SIZE}.difference(indicators.columns)
    if sales_missing:
        raise ValueError(f"Sales workbook is missing required columns: {sorted(sales_missing)}")
    if indicators_missing:
        raise ValueError(f"Indicator workbook is missing required columns: {sorted(indicators_missing)}")

    sales[DATE] = pd.to_datetime(sales[DATE])
    indicators[DATE] = pd.to_datetime(indicators[DATE])

    if sales.duplicated([BRAND, DATE]).any():
        raise ValueError("Duplicate brand-month rows found in sales file.")
    if indicators.duplicated([DATE]).any():
        raise ValueError("Duplicate monthly rows found in indicator file.")
    if (sales[TARGET] < 0).any() or (indicators[MARKET_SIZE] <= 0).any():
        raise ValueError("Raw sales must be non-negative and market_size must be positive.")

    monthly = indicators[[DATE, MARKET_SIZE]].copy()
    processed = sales[[DATE, BRAND, TARGET]].merge(monthly, on=DATE, how="left", validate="many_to_one")
    processed = processed[processed[BRAND].isin(BRANDS)].sort_values([DATE, BRAND])
    if processed[MARKET_SIZE].isna().any():
        raise ValueError("Missing market_size after merging raw files.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    return processed


def write_synthetic_sample(output_path: Path, months: int = 72) -> pd.DataFrame:
    """Write a small synthetic dataset with the same schema for smoke testing."""
    rng = np.random.default_rng(20260623)
    dates = pd.date_range("2018-01-01", periods=months, freq="MS")
    brand_weights = np.linspace(1.8, 0.35, len(BRANDS))
    brand_weights = brand_weights / brand_weights.sum()
    rows = []
    for index, date in enumerate(dates):
        seasonal = 1.0 + 0.18 * np.sin(2 * np.pi * (date.month - 1) / 12)
        market_size = int(12500 * seasonal + 30 * index + rng.normal(0, 700))
        sales = rng.multinomial(max(market_size - 1800, 1), brand_weights)
        for brand, value in zip(BRANDS, sales):
            rows.append({DATE: date, BRAND: brand, TARGET: int(value), MARKET_SIZE: market_size})
    sample = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    return sample
