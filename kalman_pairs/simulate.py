"""Fake pairs where the true hedge ratio is known, to check the filter actually works."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimulatedPair:
    dates: pd.DatetimeIndex
    x: np.ndarray
    y: np.ndarray
    alpha: np.ndarray  # true values
    beta: np.ndarray


def simulate_pair(n=1260, seed=0, alpha0=2.0, beta0=1.5, beta_vol=0.003,
                  x_vol=0.01, spread_kappa=0.15, spread_vol=0.6, start="2021-01-04"):
    # y_t = alpha + beta_t x_t + s_t
    # x is a random walk in log price, beta_t drifts, and the spread s_t mean reverts
    rng = np.random.default_rng(seed)
    x = 100 * np.exp(np.cumsum(rng.normal(0, x_vol, n)))
    beta = beta0 + np.cumsum(rng.normal(0, beta_vol, n))
    alpha = np.full(n, alpha0)

    s = np.zeros(n)
    for t in range(1, n):
        s[t] = (1 - spread_kappa) * s[t - 1] + rng.normal(0, spread_vol)

    y = alpha + beta * x + s
    dates = pd.bdate_range(start, periods=n)
    return SimulatedPair(dates, x, y, alpha, beta)
