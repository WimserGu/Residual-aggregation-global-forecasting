# Version 1 Frozen Experiment Protocol

## Research Question

Do lagged market-structure variables provide incremental predictive value for
brand-level vehicle sales beyond own-brand history, lagged macroeconomic
information, and lagged total market size?

This is a predictive study. Results must not be described as causal effects.

## Frozen Main Experiments

- E0: seasonal naive (`sales_lag_12`).
- E1: own-brand sales history, rolling means, seasonality, and brand identity.
- E2: E1 plus all predefined macroeconomic variables lagged by one month and
  lagged year-over-year BEV-share growth.
- E2.5: E2 plus official total-market sales lags 1, 2, and 3.
- E3: E2.5 plus official-denominator brand market-share lags 1, 2, and 3.
- E4: E3 plus share momentum over 1 and 12 months.
- E5: E4 plus relative share to the leading selected brand and `HHI_14`.
- E8: probabilistic E2.5.
- E9: probabilistic E5.

The primary comparison is E2.5 versus E3. E4-E5 are extensions. E8-E9 test
probabilistic forecasting value.

## Information Timing

- No random train/test split.
- Main horizon: one month ahead.
- Expanding rolling-origin evaluation with the first 48 calendar months used
  before the first test month.
- Every macroeconomic, market-size, and market-structure input is lagged.
- Market share is brand sales divided by official total-market sales.
- `HHI_14` means concentration among the 14 selected brands, not total-market
  concentration.

## Frozen Robustness Checks

- Three-step-ahead forecasting.
- Exclude Tesla.
- Large versus small brands, classified from training-period average share.
- Replace `HHI_14` with selected-brand top-three share.
- Replace lagged BEV-share growth with lagged BEV-share level.
- Log-transformed or count-oriented target specification.
- Alternative rolling test windows and deterministic model seeds.

No new feature blocks will be added in response to unfavorable main results.

