# Multi-Seed Residual Placebo Mechanism Summary

Twenty independent placebo replications were run for every K in {5, 8, 10,
12, 14}. Placebo series were regenerated inside each rolling-origin fold using
only the training-period residual values available at that origin.

Positive differences below mean that the placebo produced higher MAE than the
real residual, so the real residual was better.

| K | Placebo | Mean placebo minus real MAE | 5%-95% seed range | Seeds where real was better |
|---:|---|---:|---:|---:|
| 5 | Shuffled | 3.56 | [-7.29, 13.87] | 80% |
| 5 | Synthetic | 11.67 | [-14.93, 31.08] | 80% |
| 8 | Shuffled | 13.50 | [7.05, 19.38] | 100% |
| 8 | Synthetic | 19.34 | [8.34, 28.67] | 100% |
| 10 | Shuffled | 7.20 | [2.62, 11.73] | 95% |
| 10 | Synthetic | 11.52 | [1.32, 19.09] | 95% |
| 12 | Shuffled | 1.15 | [-1.73, 5.18] | 65% |
| 12 | Synthetic | 2.67 | [-2.25, 6.81] | 80% |
| 14 | Shuffled | 3.49 | [0.08, 6.58] | 95% |
| 14 | Synthetic | 6.50 | [2.66, 10.87] | 100% |

The results support two mechanisms. First, generic auxiliary-series pooling can
help, particularly under low coverage: shuffled and synthetic auxiliaries often
beat DirectK. Second, authentic residual dynamics add information beyond this
regularization effect at K=8, K=10, and K=14. Evidence is weaker at K=5 and
K=12.

The defensible research claim is therefore conditional: residual aggregation
can improve global forecasts through both auxiliary-task regularization and
information completion, and the balance between these channels varies with
observed market coverage.

