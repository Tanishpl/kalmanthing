"""Positions, daily P&L and the performance metrics."""

import numpy as np

TRADING_DAYS = 252


def zscore_positions(z, entry=1.0, exit=0.0):
    """Position decided at each close. +1 long the spread, -1 short, 0 flat."""
    target = np.zeros(len(z))
    pos = 0
    for t, zt in enumerate(z):
        # get out once z crosses back through the exit level
        if pos == 1 and zt >= -exit:
            pos = 0
        elif pos == -1 and zt <= exit:
            pos = 0
        # get in (or flip straight to the other side) once |z| is past entry
        if pos == 0:
            if zt > entry:
                pos = -1
            elif zt < -entry:
                pos = 1
        target[t] = pos
    return target


def backtest(y, x, target, hedge, cost_bps=5.0, lag=1):
    """Daily P&L of trading the spread with 1 unit of capital.

    target[t] is the position decided at close t and hedge[t] is the hedge ratio known then.
    With lag=1 that position is held from close t to close t+1, so a signal never earns
    the move it was calculated from.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    target = np.asarray(target, dtype=float)
    hedge = np.asarray(hedge, dtype=float)
    n = len(y)

    held = np.zeros(n)  # position held over day t, i.e. close t-1 to close t
    held[lag:] = target[: n - lag]

    pnl = np.zeros(n)
    shares_y = shares_x = 0.0
    position = 0

    for t in range(1, n):
        cost = 0.0
        if held[t] != position:
            # trade at close t-1. buy 1/G of y and sell beta/G of x where G = y + |beta| x,
            # so the whole position is worth 1
            if held[t] == 0:
                new_y = new_x = 0.0
            else:
                b = hedge[t - lag]
                G = y[t - 1] + abs(b) * x[t - 1]
                new_y = held[t] / G
                new_x = -held[t] * b / G
            traded = abs(new_y - shares_y) * y[t - 1] + abs(new_x - shares_x) * x[t - 1]
            cost = cost_bps / 10_000 * traded
            shares_y, shares_x, position = new_y, new_x, held[t]

        pnl[t] = shares_y * (y[t] - y[t - 1]) + shares_x * (x[t] - x[t - 1]) - cost

    return pnl, held


def sharpe_ratio(pnl):
    """Annualised, with a zero risk-free rate."""
    pnl = np.asarray(pnl)
    sd = pnl.std(ddof=1)
    if sd == 0:
        return 0.0
    return np.sqrt(TRADING_DAYS) * pnl.mean() / sd


def equity_curve(pnl):
    return 1 + np.cumsum(pnl)  # not compounded


def max_drawdown(pnl):
    """Biggest fall from a previous peak, as a fraction of that peak."""
    equity = np.concatenate([[1.0], equity_curve(pnl)])
    peak = np.maximum.accumulate(equity)
    return np.max((peak - equity) / peak)


def summarise(pnl, held):
    return {
        "sharpe": sharpe_ratio(pnl),
        "max_drawdown": max_drawdown(pnl),
        "total_return": pnl.sum(),
        "trades": np.count_nonzero(np.diff(held)),
        "time_in_market": np.mean(held != 0),
    }
