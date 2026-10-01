"""One-off walk-forward backtest, 2018-01-01 to 2026-09-30. Results are frozen in results/backtest/."""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_trader import data, engines, metrics, simulate  # noqa: E402
from paper_trader.config import BACKTEST_END, BACKTEST_START, DATA_START, MARKETS, START_CASH  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "backtest"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panels = {k: data.fetch(m, DATA_START).truncate(BACKTEST_END) for k, m in MARKETS.items()}
    equity, trades, summary = [], [], {"phase": "backtest", "start": BACKTEST_START, "end": BACKTEST_END,
                                       "start_cash": START_CASH, "accounts": {}}
    for name, (mkey, kind) in engines.ACCOUNTS.items():
        t0 = time.time()
        market, panel = MARKETS[mkey], panels[mkey]
        w, explain, diag = engines.targets(kind, market, panel, BACKTEST_START)
        eq, tr, _ = simulate.run(panel, w, BACKTEST_START, BACKTEST_END, START_CASH, market.cost_rate)
        tr = engines.attach_reasons(tr, panel, explain).assign(account=name)
        periods = 252 if mkey == "stocks" else 365
        s = metrics.summary(eq, tr, periods, START_CASH)
        s.update({"market": mkey, "engine": kind})
        if "allocation_history" in diag:
            diag.pop("allocation_history").to_csv(OUT / f"{name}_allocations.csv")
        s["diagnostics"] = diag
        summary["accounts"][name] = s
        equity.append(pd.DataFrame({"date": eq.index, "account": name, "equity": eq.round(2).to_numpy()}))
        trades.append(tr)
        print(f"{name}: {s['final_equity']:.2f} ({s['total_return']:+.1%}), {s['n_trades']} trades, "
              f"{time.time() - t0:.0f}s")
    pd.concat(equity).to_csv(OUT / "equity.csv", index=False)
    pd.concat(trades).round(6).to_csv(OUT / "trades.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
