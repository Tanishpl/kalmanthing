"""
python -m kalman_pairs                    # KO/PEP, XOM/CVX, V/MA from 2021 to 2025
python -m kalman_pairs --pairs KO:PEP     # just one pair, KO regressed on PEP
python -m kalman_pairs --simulate         # fake pairs, no download
"""

import argparse
from pathlib import Path

from .data import load_closes
from .pipeline import Config, run_pair
from .plots import plot_pair
from .simulate import simulate_pair


def results_table(rows, cfg, title):
    lines = [
        f"# {title}",
        "",
        f"Trained on the first {cfg.train_years} year, traded the rest out of sample. "
        f"Entry at |z| > {cfg.entry_z:g}, exit when z crosses {cfg.exit_z:g}, "
        f"{cfg.lag} bar lag, {cfg.cost_bps:g} bps per trade.",
        "",
        "| Pair | Strategy | Sharpe | Max drawdown | Return | Return before costs | Trades | Time in market |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for name, strat in [("Kalman", r.kalman), ("Static OLS", r.static)]:
            m = strat.metrics
            lines.append(
                f"| {r.y_name}/{r.x_name} | {name} | {m['sharpe']:.2f} | {m['max_drawdown']:.1%} | "
                f"{m['total_return']:+.1%} | {m['gross_return']:+.1%} | {m['trades']} | {m['time_in_market']:.0%} |"
            )

    lines += [
        "",
        "Returns are on 1 unit of capital, not compounded. Trades counts position changes.",
        "",
        "Fitted on the training year",
        "",
        "| Pair | delta | R | Static OLS beta |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r.y_name}/{r.x_name} | {r.params.delta:.1e} | {r.params.R:.4g} | {r.ols_beta:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="Kalman filter pairs trading backtest")
    p.add_argument("--pairs", nargs="+", default=["KO:PEP", "XOM:CVX", "V:MA"], help="Y:X, Y is regressed on X")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2026-01-01", help="exclusive")
    p.add_argument("--train-years", type=int, default=1)
    p.add_argument("--entry", type=float, default=1.0)
    p.add_argument("--exit", type=float, default=0.0)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--lag", type=int, default=1, choices=[1, 2])
    p.add_argument("--simulate", action="store_true", help="use fake pairs instead of downloading")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    cfg = Config(args.train_years, args.entry, args.exit, args.cost_bps, args.lag)
    out = Path(args.out) / "simulated" if args.simulate else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.simulate:
        for seed in range(3):
            sim = simulate_pair(seed=seed)
            r = run_pair(sim.dates, sim.y, sim.x, cfg, f"SIM{seed}Y", f"SIM{seed}X")
            plot_pair(r, cfg, out / f"sim_{seed}.png", true_beta=sim.beta)
            rows.append(r)
        title = "Simulated pairs"
    else:
        for pair in args.pairs:
            y_name, x_name = pair.split(":")
            closes = load_closes(y_name, x_name, args.start, args.end)
            r = run_pair(closes.index, closes[y_name].values, closes[x_name].values, cfg, y_name, x_name)
            plot_pair(r, cfg, out / f"{y_name}_{x_name}.png")
            rows.append(r)
        title = f"Results, {args.start} to {args.end}"

    table = results_table(rows, cfg, title)
    (out / "results.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
