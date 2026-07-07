# Raw Data

Raw data are not redistributed in this repository. The manuscript uses monthly
Norwegian passenger-vehicle registration data from **OFV Registreringsstatistikk**
and official monthly total market size for the same months.

## Study Period

```text
2018-01 to 2026-03
```

## Required Raw Files

To reproduce the manuscript numerically, place licensed copies of the following
files in this folder:

```text
data/raw/Norway_monthly_car_sale.xlsx
data/raw/All_indicators.xlsx
```

Expected workbook structure:

| File | Sheet | Required columns | Notes |
| --- | --- | --- | --- |
| `Norway_monthly_car_sale.xlsx` | `Sheet2` | `Brand`, `Sales`, `date` | One row per observed brand-month. |
| `All_indicators.xlsx` | `Ark1` | `Month`, `market_size` | One row per month. |

Only `market_size` is required from `All_indicators.xlsx` for the final
residual-aggregation experiments.

## Access Procedure

1. Obtain monthly Norwegian passenger-vehicle registration data from OFV
   Registreringsstatistikk or an equivalent licensed historical source.
2. Extract monthly registrations for the 14 observed manuscript brands:
   `Audi`, `BMW`, `Ford`, `Hyundai`, `Kia`, `Mazda`, `Mercedes-Benz`, `Nissan`,
   `Peugeot`, `Skoda`, `Tesla`, `Toyota`, `Volkswagen`, and `Volvo`.
3. Obtain official monthly total passenger-vehicle registrations for the same
   months.
4. Save the two workbooks using the filenames and sheet names listed above.
5. Run:

```bash
python scripts/prepare_processed_data.py
```

This creates:

```text
data/processed/monthly_market_series.csv
```

## Provenance and Redistribution

- Provider: OFV Registreringsstatistikk / equivalent Norwegian registration source.
- Access date: the historical dataset was assembled from repeated downloads of
  OFV registration statistics during the course of this research project rather
  than obtained through a single download. Accordingly, no single access date
  applies to the complete dataset.
- Redistribution: not assumed. Confirm the data provider's licensing terms before
  adding raw or full processed data to a public repository.

## Synthetic Alternative

If licensed historical data are unavailable, generate a schema-compatible
synthetic sample for smoke testing:

```bash
python scripts/prepare_processed_data.py --synthetic-sample --output data/processed/monthly_market_series.csv
```

Synthetic data do not reproduce the manuscript's numerical results.
