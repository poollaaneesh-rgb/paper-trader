import numpy as np
import pandas as pd

from paper_trader import metrics, simulate
from paper_trader.portfolio import Account
from tests.conftest import make_panel


def bench_weights(panel, ticker, decide_on):
    w = pd.DataFrame(np.nan, index=panel.dates, columns=panel.tickers)
    w.loc[decide_on] = 0.0
    w.loc[decide_on, ticker] = 1.0
    return w


def test_benchmark_trades_once(panel):
    start = panel.dates[10]
    w = bench_weights(panel, "AAA", panel.dates[9])
    eq, trades, _ = simulate.run(panel, w, start, panel.dates[-1], 400.0, 0.0005)
    assert len(trades) == 1
    assert trades.iloc[0]["date"] == start


def test_equity_is_cash_plus_marked_positions(panel):
    w = pd.DataFrame(0.0, index=panel.dates, columns=panel.tickers)
    w["AAA"], w["BBB"] = 0.5, 0.5
    eq, _, acct = simulate.run(panel, w, panel.dates[5], panel.dates[50], 400.0, 0.001)
    last = panel.close.loc[panel.dates[50]]
    assert np.isclose(eq.iloc[-1], acct.cash + sum(u * last[t] for t, u in acct.positions.items()))


def test_fill_happens_at_next_open(panel):
    # Perfect foresight of the close-to-close move cannot be captured: orders fill at the next open.
    w = pd.DataFrame(0.0, index=panel.dates, columns=panel.tickers)
    eq, trades, _ = simulate.run(panel, w.assign(AAA=1.0), panel.dates[5], panel.dates[6], 400.0, 0.0)
    assert trades.iloc[0]["price"] == panel.open.loc[panel.dates[5], "AAA"]


def test_step_holds_when_target_is_none(panel):
    acct = Account(cash=400.0)
    trades = simulate.step(acct, panel.dates[5], panel, None, 0.001)
    assert trades == [] and acct.cash == 400.0


def test_metrics_summary():
    eq = pd.Series([400, 440, 396, 480.0], index=pd.date_range("2024-01-01", periods=4))
    trades = pd.DataFrame({"realized_pnl": [5.0, -2.0, np.nan]})
    s = metrics.summary(eq, trades, periods=252, start_cash=400.0)
    assert np.isclose(s["total_return"], 0.2)
    assert np.isclose(s["max_drawdown"], 396 / 440 - 1)
    assert s["win_rate"] == 0.5
    assert s["n_trades"] == 3
