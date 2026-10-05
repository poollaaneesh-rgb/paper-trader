"""One-off walk-forward backtest, 2018-01-01 to 2026-09-30. Results are frozen in results/backtest/.

Each self-tuning account is simulated twice: as it runs (following its menu) and on its original fixed rules, so
the two can be compared on the same prices. The model's odds are saved for scripts/controls.py."""

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_trader import data, engines, metrics, simulate
from paper_trader.config import BACKTEST_END, BACKTEST_START, DATA_START, MARKETS, START_CASH

OUT = Path(__file__).resolve().parents[1] / "results" / "backtest"


def frozen_panel(key, market):
    """The backtest's prices are saved once and reused: Yahoo's adjusted prices shift by about 1e-6
    between downloads, which is enough to change the ML results. A frozen copy makes them reproducible."""
    path = OUT / f"prices_{key}.csv.gz"
    if path.exists():
        df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
        return data.Panel(*(df[f].astype(float) for f in ("open", "close", "volume")))
    panel = data.fetch(market, DATA_START).truncate(BACKTEST_END)
    pd.concat({"open": panel.open, "close": panel.close, "volume": panel.volume}, axis=1).to_csv(path)
    return panel


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panels = {k: frozen_panel(k, m) for k, m in MARKETS.items()}
    equity, trades, summary = (
        [],
        [],
        {"phase": "backtest", "start": BACKTEST_START, "end": BACKTEST_END, "start_cash": START_CASH, "accounts": {}},
    )
    for name, (mkey, kind) in engines.ACCOUNTS.items():
        t0 = time.time()
        market, panel = MARKETS[mkey], panels[mkey]
        periods = 252 if mkey == "stocks" else 365
        w, explain, diag = engines.targets(kind, market, panel, BACKTEST_START)
        runs = [(name, kind, w, explain)]
        if "fixed_weights" in diag:
            runs.append(
                (
                    f"{name}_fixed",
                    f"{kind}_fixed",
                    diag.pop("fixed_weights"),
                    lambda d, t, side: "the original fixed rules",
                )
            )
        for acct, engine, weights, why in runs:
            eq, tr, _ = simulate.run(panel, weights, BACKTEST_START, BACKTEST_END, START_CASH, market.cost_rate)
            tr = engines.attach_reasons(tr, panel, why).assign(account=acct)
            s = metrics.summary(eq, tr, periods, START_CASH)
            s.update({"market": mkey, "engine": engine})
            summary["accounts"][acct] = s
            equity.append(pd.DataFrame({"date": eq.index, "account": acct, "equity": eq.round(2).to_numpy()}))
            trades.append(tr)
            took = f"{time.time() - t0:.0f}s"
            print(f"{acct}: {s['final_equity']:.2f} ({s['total_return']:+.1%}), {s['n_trades']} trades, {took}")
        if "chosen" in diag:
            chosen = diag.pop("chosen").loc[BACKTEST_START:BACKTEST_END]
            chosen.rename("variant").to_csv(OUT / f"{name}_variants.csv", index_label="date")
            diag["time_in_variants"] = {
                k: round(float(v), 3) for k, v in chosen.value_counts(normalize=True).head(5).items()
            }
        diag.pop("scores", None)
        if "probs" in diag:
            diag.pop("probs").to_csv(OUT / f"ml_probs_{mkey}.csv.gz")
        summary["accounts"][name]["diagnostics"] = diag
    pd.concat(equity).to_csv(OUT / "equity.csv", index=False)
    # Only the fixed accounts' equity curves are shown, so their trades stay out of the file to keep it lean.
    all_trades = pd.concat(trades)
    all_trades[~all_trades["account"].str.endswith("_fixed")].to_csv(
        OUT / "trades.csv", index=False, float_format="%.6f"
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
