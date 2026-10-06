"""One figure per pair. Hedge ratio, z-score and out of sample equity."""

import matplotlib

matplotlib.use("Agg")  # just save to file, no window
import matplotlib.pyplot as plt

from .backtest import equity_curve


def plot_pair(result, cfg, path, true_beta=None):
    d = result.dates
    s = result.split
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    if true_beta is not None:
        ax.plot(d, true_beta, color="black", lw=1.5, label="True (simulated)")
    ax.plot(d, result.kf.beta, color="C0", lw=2, label="Kalman")
    ax.axhline(result.ols_beta, color="C1", lw=2, label="Static OLS")
    ax.set_title(f"Hedge ratio, {result.y_name} on {result.x_name}", loc="left")
    ax.legend()

    ax = axes[1]
    ax.plot(d, result.kalman.z, color="C0", lw=0.8)
    ax.axhline(cfg.entry_z, color="grey", ls="--")
    ax.axhline(-cfg.entry_z, color="grey", ls="--")
    ax.set_ylim(-5, 5)
    ax.set_title("Kalman z-score e / sqrt(Q)", loc="left")

    ax = axes[2]
    ax.plot(d[s:], equity_curve(result.kalman.pnl[s:]), color="C0", lw=2,
            label=f"Kalman, Sharpe {result.kalman.metrics['sharpe']:.2f}")
    ax.plot(d[s:], equity_curve(result.static.pnl[s:]), color="C1", lw=2,
            label=f"Static OLS, Sharpe {result.static.metrics['sharpe']:.2f}")
    ax.axhline(1, color="grey", lw=1)
    ax.set_title(f"Out of sample equity, {cfg.cost_bps:g} bps per trade", loc="left")
    ax.legend()

    # grey = training year
    for ax in axes:
        ax.axvspan(d[0], d[s], color="grey", alpha=0.1)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
