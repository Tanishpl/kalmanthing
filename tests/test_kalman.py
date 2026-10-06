import numpy as np

from kalman_pairs.kalman import fit_noise, kalman_regression, ols
from kalman_pairs.simulate import simulate_pair


def test_finds_a_fixed_line_from_a_bad_start():
    # y = 2 + 1.5x + noise, starting the filter at alpha = beta = 0
    rng = np.random.default_rng(1)
    n = 2000
    x = 50 + np.cumsum(rng.normal(0, 1, n))
    y = 2 + 1.5 * x + rng.normal(0, 1, n)

    kf = kalman_regression(x, y, delta=1e-8, R=1.0, theta0=[0, 0], P0=np.eye(2) * 1e3)
    assert abs(kf.beta[-1] - 1.5) < 0.01
    assert abs(kf.alpha[-1] - 2.0) < 0.5


def test_forecast_only_uses_the_past():
    rng = np.random.default_rng(2)
    x = 50 + np.cumsum(rng.normal(0, 1, 100))
    y = 1 + 2 * x + rng.normal(0, 1, 100)
    kf = kalman_regression(x, y, 1e-4, 1.0, [0, 0], np.eye(2) * 10)

    # e_t is the error of the forecast made before seeing y_t
    np.testing.assert_allclose(kf.e, y - (kf.alpha_pred + kf.beta_pred * x))
    # and that forecast uses yesterday's update
    np.testing.assert_allclose(kf.beta_pred[1:], kf.beta[:-1])


def test_tracks_a_drifting_hedge_ratio_better_than_ols():
    sim = simulate_pair(seed=0, beta_vol=0.003)
    split = 261
    params = fit_noise(sim.x[:split], sim.y[:split])
    kf = kalman_regression(sim.x, sim.y, params.delta, params.R, params.theta0, params.P0)
    _, b, _, _ = ols(sim.x[:split], sim.y[:split])

    kalman_error = np.sqrt(np.mean((kf.beta[split:] - sim.beta[split:]) ** 2))
    ols_error = np.sqrt(np.mean((b - sim.beta[split:]) ** 2))
    assert kalman_error < 0.5 * ols_error


def test_max_likelihood_finds_the_true_noise():
    # simulate straight from the model with known delta and R, fit should land
    # within one grid step of the truth
    rng = np.random.default_rng(3)
    n, true_delta, true_R = 1500, 1e-5, 0.5
    w_sd = np.sqrt(true_delta / (1 - true_delta))
    x = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    alpha = 2 + np.cumsum(rng.normal(0, w_sd, n))
    beta = 1.5 + np.cumsum(rng.normal(0, w_sd, n))
    y = alpha + beta * x + rng.normal(0, np.sqrt(true_R), n)

    params = fit_noise(x, y)
    assert abs(np.log10(params.delta) - np.log10(true_delta)) <= 0.5 + 1e-9
    assert 0.5 * true_R < params.R < 2 * true_R
