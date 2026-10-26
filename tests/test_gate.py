import numpy as np
import pandas as pd

from paper_trader import engines
from paper_trader.config import CRYPTO_GATE, Market
from tests.conftest import make_panel


def _crypto_panel(n_days=500):
    return make_panel(n_days=n_days, tickers=("BTC-USD", "ETH-USD", "SOL-USD"), freq="D")


def test_gate_reads_no_future_closes():
    p = _crypto_panel()
    full = engines.trend_gate(p, "BTC-USD", 100, 0.03)
    for t in p.dates[[150, 300, 499]]:
        part = engines.trend_gate(p.truncate(t), "BTC-USD", 100, 0.03)
        assert part.loc[t] == full.loc[t]


def test_gate_holds_its_state_inside_the_band():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    close = pd.DataFrame({"BTC-USD": [100, 100, 110, 104, 104, 90]}, index=dates, dtype=float)
    p = type(make_panel())(open=close, close=close, volume=close)
    g = engines.trend_gate(p, "BTC-USD", 2, 0.03)
    # 2-day averages: -, 100, 105, 107, 104, 97; in at 110 > 108.15, held at 104 (inside 3%), out at 90 < 94.09
    assert g.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]


def test_crypto_engines_hold_nothing_while_bitcoin_is_in_a_downtrend():
    p = _crypto_panel(800)
    mkt = Market("crypto", tuple(p.tickers), "BTC-USD", 0.003, 2)
    gate = engines.trend_gate(p, "BTC-USD", CRYPTO_GATE["days"], CRYPTO_GATE["band"])
    w, explain, _ = engines.targets("tournament", mkt, p, p.dates[400])
    off = gate.index[gate == 0.0]
    assert (w.loc[off].fillna(0.0) == 0.0).all().all()
    assert "downtrend" in explain(off[-1], "ETH-USD", "sell")
    ungated, _, _ = engines._targets("tournament", mkt, p, p.dates[400])
    on = gate.index[gate == 1.0]
    pd.testing.assert_frame_equal(w.loc[on], ungated.loc[on])


def test_stock_engines_are_not_gated(panel):
    mkt = Market("stocks", tuple(panel.tickers), "AAA", 0.0005, 2)
    w, _, diag = engines.targets("tournament", mkt, panel, panel.dates[300])
    ungated, _, _ = engines._targets("tournament", mkt, panel, panel.dates[300])
    pd.testing.assert_frame_equal(w, ungated)
    assert "gate" not in diag and np.isfinite(w.fillna(0).to_numpy()).all()
