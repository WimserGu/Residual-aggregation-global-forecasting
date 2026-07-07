# Processed data

Main reproduction scripts read:

```text
data/processed/monthly_market_series.csv
```

Required schema:

| Column | Type | Description |
| --- | --- | --- |
| `date` | date | Monthly timestamp. |
| `brand` | string | One of the 14 observed manuscript brands. |
| `sales` | numeric | Brand-month passenger-vehicle registrations. |
| `market_size` | numeric | Official total market registrations for the same month. |

The file must contain a balanced brand-month panel for the 14 observed brands:

`Audi`, `BMW`, `Ford`, `Hyundai`, `Kia`, `Mazda`, `Mercedes-Benz`, `Nissan`,
`Peugeot`, `Skoda`, `Tesla`, `Toyota`, `Volkswagen`, and `Volvo`.

The residual category is constructed inside the experiment code as:

```text
Residual_t = market_size_t - sum(selected_brand_sales_t)
```

The public repository includes a synthetic sample file for smoke testing only.
It should not be used to reproduce the manuscript's numerical results.

## Creating the Processed File

With licensed raw files in `data/raw/`, run:

```bash
python scripts/prepare_processed_data.py
```

The script expects:

| Raw file | Sheet | Input columns | Processed columns |
| --- | --- | --- | --- |
| `data/raw/Norway_monthly_car_sale.xlsx` | `Sheet2` | `Brand`, `Sales`, `date` | `brand`, `sales`, `date` |
| `data/raw/All_indicators.xlsx` | `Ark1` | `Month`, `market_size` | `date`, `market_size` |

The output is:

```text
data/processed/monthly_market_series.csv
```
