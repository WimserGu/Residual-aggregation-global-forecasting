"""Prepare processed data or create a synthetic sample.

This script converts the original Excel files to the processed CSV expected by
the reproduction scripts. If raw files cannot be redistributed, use
``--synthetic-sample`` to create a small schema-compatible sample for smoke
testing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from residual_aggregation.data import prepare_from_excel, write_synthetic_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales-xlsx", type=Path, default=Path("data/raw/Norway_monthly_car_sale.xlsx"))
    parser.add_argument("--indicators-xlsx", type=Path, default=Path("data/raw/All_indicators.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/monthly_market_series.csv"))
    parser.add_argument("--synthetic-sample", action="store_true")
    args = parser.parse_args()

    if args.synthetic_sample:
        frame = write_synthetic_sample(args.output)
        print(f"Wrote synthetic sample with {len(frame)} rows to {args.output}")
        return

    frame = prepare_from_excel(args.sales_xlsx, args.indicators_xlsx, args.output)
    print(f"Wrote processed data with {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
