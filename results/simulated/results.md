# Simulated pairs

Trained on the first 1 year, traded the rest out of sample. Entry at |z| > 1, exit when z crosses 0, 1 bar lag, 5 bps per trade.

| Pair | Strategy | Sharpe | Max drawdown | Return | Return before costs | Trades | Time in market |
|---|---|---|---|---|---|---|---|
| SIM0Y/SIM0X | Kalman | -0.29 | 7.3% | -3.8% | +20.2% | 396 | 53% |
| SIM0Y/SIM0X | Static OLS | 1.15 | 2.3% | +13.8% | +14.8% | 20 | 60% |
| SIM1Y/SIM1X | Kalman | 0.43 | 5.4% | +7.0% | +33.1% | 413 | 62% |
| SIM1Y/SIM1X | Static OLS | 0.00 | 6.2% | +0.0% | +0.5% | 9 | 96% |
| SIM2Y/SIM2X | Kalman | -0.47 | 10.6% | -7.3% | +16.3% | 385 | 59% |
| SIM2Y/SIM2X | Static OLS | 0.41 | 3.9% | +6.7% | +7.7% | 21 | 82% |

Returns are on 1 unit of capital, not compounded. Trades counts position changes.

Fitted on the training year

| Pair | delta | R | Static OLS beta |
|---|---|---|---|
| SIM0Y/SIM0X | 3.2e-05 | 0.08775 | 1.745 |
| SIM1Y/SIM1X | 3.2e-05 | 0.0788 | 1.664 |
| SIM2Y/SIM2X | 3.2e-05 | 0.08645 | 1.809 |
