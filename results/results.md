# Results, 2021-01-01 to 2026-01-01

Trained on the first 1 year, traded the rest out of sample. Entry at |z| > 1, exit when z crosses 0, 1 bar lag, 5 bps per trade.

| Pair | Strategy | Sharpe | Max drawdown | Return | Return before costs | Trades | Time in market |
|---|---|---|---|---|---|---|---|
| KO/PEP | Kalman | -1.42 | 34.2% | -32.3% | -9.2% | 363 | 54% |
| KO/PEP | Static OLS | -0.38 | 24.0% | -11.9% | -11.4% | 11 | 90% |
| XOM/CVX | Kalman | -1.77 | 35.7% | -35.3% | -15.6% | 334 | 41% |
| XOM/CVX | Static OLS | -0.33 | 18.4% | -10.1% | -9.7% | 7 | 99% |
| V/MA | Kalman | -0.31 | 10.2% | -4.9% | +11.8% | 286 | 36% |
| V/MA | Static OLS | 1.19 | 3.4% | +20.5% | +21.5% | 21 | 58% |

Returns are on 1 unit of capital, not compounded. Trades counts position changes.

Fitted on the training year

| Pair | delta | R | Static OLS beta |
|---|---|---|---|
| KO/PEP | 3.2e-06 | 0.01536 | 0.224 |
| XOM/CVX | 3.2e-05 | 0.003572 | 0.594 |
| V/MA | 1.0e-05 | 0.5806 | 0.566 |
