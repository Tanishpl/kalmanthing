import numpy as np
import pytest

from kalman_pairs.backtest import backtest, max_drawdown, sharpe_ratio, zscore_positions


def test_position_rules():
    z = [0, 1.5, 0.5, -0.1, -1.5, 2.0, 0.0]
    # flat, go short, hold, exit at 0, go long, flip to short, exit
    assert list(zscore_positions(z, entry=1.0, exit=0.0)) == [0, -1, -1, 0, 1, -1, 0]


# small example worked out by hand
#   signal at close 1 says go long. G = 10 + 1*5 = 15 so hold 1/15 of y and -1/15 of x
#   day 2  y +1, x 0   ->  pnl 1/15
#   day 3  y +1, x +1  ->  pnl 0
#   signal at close 3 says flat, so day 4 earns nothing
Y = [10, 10, 11, 12, 12]
X = [5, 5, 5, 6, 5]
TARGET = [0, 1, 1, 0, 0]
HEDGE = [1, 1, 1, 1, 1]


def test_pnl_by_hand():
    pnl, held, costs = backtest(Y, X, TARGET, HEDGE, cost_bps=0, lag=1)
    assert list(held) == [0, 0, 1, 1, 0]
    np.testing.assert_allclose(pnl, [0, 0, 1 / 15, 0, 0], atol=1e-12)
    assert costs.sum() == 0


def test_pnl_by_hand_with_costs():
    # entry trades 10/15 + 5/15 = 1.0 of notional, exit trades 12/15 + 6/15 = 1.2
    # so at 5bps that's 0.0005 then 0.0006
    pnl, _, costs = backtest(Y, X, TARGET, HEDGE, cost_bps=5, lag=1)
    np.testing.assert_allclose(costs, [0, 0, 0.0005, 0, 0.0006], atol=1e-12)
    np.testing.assert_allclose(pnl, [0, 0, 1 / 15 - 0.0005, 0, -0.0006], atol=1e-12)


def test_lag_two():
    _, held, _ = backtest(Y, X, TARGET, HEDGE, cost_bps=0, lag=2)
    assert list(held) == [0, 0, 0, 1, 1]


def test_signal_cant_earn_its_own_day():
    # a "signal" that knows whether y went up today. without the lag this has a
    # sharpe of about 20, with it there's nothing to earn
    rng = np.random.default_rng(0)
    y = 100 + np.cumsum(rng.normal(0, 1, 500))
    x = np.full(500, 50.0)
    cheat = np.sign(np.diff(y, prepend=y[0]))
    pnl, _, _ = backtest(y, x, cheat, np.zeros(500), cost_bps=0, lag=1)
    assert abs(sharpe_ratio(pnl)) < 1.0


def test_max_drawdown():
    # equity 1 -> 1.1 -> 0.88 -> 0.93, worst fall is 0.22 from 1.1
    assert max_drawdown([0.1, -0.22, 0.05]) == pytest.approx(0.2)


def test_sharpe_of_nothing_is_zero():
    assert sharpe_ratio(np.zeros(10)) == 0.0
