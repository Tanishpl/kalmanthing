"""Runs the Kalman strategy and the static OLS baseline on one pair."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import backtest, summarise, zscore_positions
from .kalman import KalmanParams, KalmanResult, fit_noise, kalman_regression, ols


@dataclass
class Config:
    train_years: int = 1
    entry_z: float = 1.0
    exit_z: float = 0.0
    cost_bps: float = 5.0
    lag: int = 1


@dataclass
class StrategyResult:
    z: np.ndarray
    hedge: np.ndarray
    target: np.ndarray
    held: np.ndarray
    pnl: np.ndarray
    metrics: dict  # test period only


@dataclass
class PairResult:
    y_name: str
    x_name: str
    dates: pd.DatetimeIndex
    split: int  # first day of the test period
    params: KalmanParams
    kf: KalmanResult
    ols_alpha: float
    ols_beta: float
    kalman: StrategyResult
    static: StrategyResult


def split_index(dates, train_years):
    """Index of the first day after the training period."""
    dates = pd.DatetimeIndex(dates)
    cutoff = dates[0] + pd.DateOffset(years=train_years)
    split = int((dates < cutoff).sum())
    if split <= 10 or split >= len(dates) - 10:
        raise ValueError("not enough data either side of the train/test split")
    return split


def trade(y, x, z, hedge, split, cfg):
    target = zscore_positions(z, cfg.entry_z, cfg.exit_z)
    target[:split] = 0  # no trading in the training year
    pnl, held = backtest(y, x, target, hedge, cfg.cost_bps, cfg.lag)
    metrics = summarise(pnl[split:], held[split:])
    return StrategyResult(z, hedge, target, held, pnl, metrics)


def run_pair(dates, y, x, cfg=Config(), y_name="Y", x_name="X"):
    """Fit everything on the first year, then trade the rest out of sample."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    split = split_index(dates, cfg.train_years)

    # kalman. runs over the whole series but only ever uses data up to day t,
    # so the training year just warms it up
    params = fit_noise(x[:split], y[:split])
    kf = kalman_regression(x, y, params.delta, params.R, params.theta0, params.P0)
    z_kalman = kf.e / np.sqrt(kf.Q)
    kalman = trade(y, x, z_kalman, kf.beta, split, cfg)

    # static ols baseline, one line fitted on the training year
    a, b, resid_var, _ = ols(x[:split], y[:split])
    z_static = (y - a - b * x) / np.sqrt(resid_var)
    static = trade(y, x, z_static, np.full(len(y), b), split, cfg)

    return PairResult(y_name, x_name, pd.DatetimeIndex(dates), split, params, kf, a, b, kalman, static)
