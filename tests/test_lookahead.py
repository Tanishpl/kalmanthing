# if there's no lookahead bias, changing prices after day T can't change anything the
# strategy did up to day T. so scramble the future and check the past is identical

import numpy as np

from kalman_pairs.kalman import kalman_regression, ols
from kalman_pairs.pipeline import Config, run_pair
from kalman_pairs.simulate import simulate_pair

T = 700  # somewhere in the test period


def scrambled_future(sim, seed=99):
    rng = np.random.default_rng(seed)
    y, x = sim.y.copy(), sim.x.copy()
    y[T + 1:] = y[T + 1:] * 1.3 + rng.normal(0, 5, len(y) - T - 1)
    x[T + 1:] = x[T + 1:] * 0.8
    return y, x


def test_changing_the_future_doesnt_change_the_past():
    sim = simulate_pair(seed=4)
    y2, x2 = scrambled_future(sim)

    a = run_pair(sim.dates, sim.y, sim.x, Config())
    b = run_pair(sim.dates, y2, x2, Config())

    # same fitted parameters
    assert a.params.delta == b.params.delta and a.params.R == b.params.R
    assert a.ols_beta == b.ols_beta

    # same signals, positions and pnl up to day T, for both strategies
    upto = slice(0, T + 1)
    for sa, sb in [(a.kalman, b.kalman), (a.static, b.static)]:
        np.testing.assert_array_equal(sa.z[upto], sb.z[upto])
        np.testing.assert_array_equal(sa.target[upto], sb.target[upto])
        np.testing.assert_array_equal(sa.pnl[upto], sb.pnl[upto])
    np.testing.assert_array_equal(a.kf.beta[upto], b.kf.beta[upto])

    # make sure the scramble actually changed something afterwards
    assert not np.array_equal(a.kalman.z[T + 1:], b.kalman.z[T + 1:])


def test_it_catches_the_v1_bug():
    # v1 set the noise from the variance of the whole sample. doing that here should
    # change the early signals when the future changes, otherwise the test is useless
    sim = simulate_pair(seed=4)
    y2, x2 = scrambled_future(sim)

    def leaky_z(y, x):
        a, b, _, cov = ols(x[:261], y[:261])
        R = np.var(y - a - b * x)  # whole sample, the v1 mistake
        kf = kalman_regression(x, y, 1e-5, R, [a, b], cov)
        return kf.e / np.sqrt(kf.Q)

    assert not np.array_equal(leaky_z(sim.y, sim.x)[: T + 1], leaky_z(y2, x2)[: T + 1])
