"""Kalman filter for a regression line y = alpha + beta * x that drifts over time."""

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanResult:
    alpha: np.ndarray       # estimates after seeing y_t
    beta: np.ndarray
    alpha_pred: np.ndarray  # estimates before seeing y_t, what the forecast used
    beta_pred: np.ndarray
    e: np.ndarray           # forecast error
    Q: np.ndarray           # variance of the forecast error
    loglik: float


def kalman_regression(x, y, delta, R, theta0, P0):
    # state is theta = [alpha, beta] and both are random walks
    #   theta_t = theta_{t-1} + w_t          w_t ~ N(0, W),  W = delta / (1 - delta) * I
    #   y_t = alpha_t + beta_t * x_t + eps_t  eps_t ~ N(0, R)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    W = delta / (1 - delta) * np.eye(2)
    theta = np.array(theta0, dtype=float)
    P = np.array(P0, dtype=float)

    alpha, beta = np.zeros(n), np.zeros(n)
    alpha_pred, beta_pred = np.zeros(n), np.zeros(n)
    e_all, Q_all = np.zeros(n), np.zeros(n)

    for t in range(n):
        # predict. random walk so the best guess doesn't change, we just get less sure
        theta_pred = theta
        P_pred = P + W

        # forecast y_t from x_t before looking at y_t
        H = np.array([1.0, x[t]])
        e = y[t] - H @ theta_pred
        Q = H @ P_pred @ H + R

        # update. K is the kalman gain, how far this error moves alpha and beta
        K = P_pred @ H / Q
        theta = theta_pred + K * e
        P = P_pred - np.outer(K, H @ P_pred)

        alpha_pred[t], beta_pred[t] = theta_pred
        alpha[t], beta[t] = theta
        e_all[t], Q_all[t] = e, Q

    # gaussian log likelihood of the forecast errors, used to choose delta and R
    loglik = -0.5 * np.sum(np.log(2 * np.pi * Q_all) + e_all**2 / Q_all)
    return KalmanResult(alpha, beta, alpha_pred, beta_pred, e_all, Q_all, float(loglik))


def ols(x, y):
    """Fit y = a + b x. Returns a, b, the residual variance and the covariance of (a, b)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coef
    resid_var = resid @ resid / (len(y) - 2)
    cov = resid_var * np.linalg.inv(X.T @ X)
    return coef[0], coef[1], resid_var, cov


@dataclass
class KalmanParams:
    delta: float
    R: float
    theta0: np.ndarray
    P0: np.ndarray
    loglik: float


def fit_noise(x_train, y_train):
    """Pick delta and R by maximum likelihood, using the training data only.

    v1 set the noise from the variance of the whole sample, which was lookahead bias.
    The filter starts from the OLS fit on the training data, and P0 is that fit's
    covariance, so it starts exactly as unsure as OLS says it should be.
    """
    a, b, resid_var, cov = ols(x_train, y_train)
    theta0 = np.array([a, b])

    best = None
    for delta in np.logspace(-8, -2, 13):
        for scale in np.logspace(-3, 0.5, 15):
            R = resid_var * scale
            ll = kalman_regression(x_train, y_train, delta, R, theta0, cov).loglik
            if best is None or ll > best.loglik:
                best = KalmanParams(delta, R, theta0, cov, ll)
    return best
