import copy

import pandas as pd

from paper_trader import live
from paper_trader.config import Market
from tests.conftest import make_panel

MKT = {"crypto": Market("crypto", ("AAA", "BBB", "CCC", "DDD"), "AAA", 0.005, 2)}
ACCTS = {"tournament_crypto": ("crypto", "tournament"), "bench_btc": ("crypto", "bench")}



def test_initialises_and_is_idempotent(monkeypatch):
    monkeypatch.setattr(live, "LIVE_START", "2026-10-01")
    p = make_panel(n_days=2650, freq="D")  # 2020-01-01 .. 2027-04
    p = p.truncate("2026-10-05")
    today = pd.Timestamp("2026-10-06")
    state = live.init_state(ACCTS)
    state, eq, trades = live.run(state, {"crypto": p}, today, MKT, ACCTS)
    assert {r["account"] for r in eq} == set(ACCTS)
    assert len([r for r in eq if r["account"] == "bench_btc"]) == 5
    bench_trades = [t for t in trades if t["account"] == "bench_btc"]
    assert len(bench_trades) == 1 and str(bench_trades[0]["date"].date()) == "2026-10-01"
    snapshot = copy.deepcopy(state)
    state, eq2, trades2 = live.run(state, {"crypto": p}, today, MKT, ACCTS)
    assert eq2 == [] and trades2 == [] and state == snapshot


def test_stale_market_is_skipped():
    p = make_panel(n_days=2650, freq="D").truncate("2026-10-01")
    state = live.init_state(ACCTS)
    before = copy.deepcopy(state["accounts"])
    state, eq, trades = live.run(state, {"crypto": p}, pd.Timestamp("2026-10-10"), MKT, ACCTS)
    assert eq == [] and trades == []
    assert state["accounts"] == before
    assert state["skips"][-1]["market"] == "crypto"
