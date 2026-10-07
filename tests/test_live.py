import copy

import pandas as pd

from paper_trader import engines, live
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
    state, eq, trades, _ = live.run(state, {"crypto": p}, today, MKT, ACCTS)
    assert {r["account"] for r in eq} == set(ACCTS)
    assert len([r for r in eq if r["account"] == "bench_btc"]) == 5
    bench_trades = [t for t in trades if t["account"] == "bench_btc"]
    assert len(bench_trades) == 1 and str(bench_trades[0]["date"].date()) == "2026-10-01"
    snapshot = copy.deepcopy(state)
    state, eq2, trades2, _ = live.run(state, {"crypto": p}, today, MKT, ACCTS)
    assert eq2 == [] and trades2 == [] and state == snapshot


def test_stale_market_is_skipped():
    p = make_panel(n_days=2650, freq="D").truncate("2026-10-01")
    state = live.init_state(ACCTS)
    before = copy.deepcopy(state["accounts"])
    state, eq, trades, _ = live.run(state, {"crypto": p}, pd.Timestamp("2026-10-10"), MKT, ACCTS)
    assert eq == [] and trades == []
    assert state["accounts"] == before
    assert state["skips"][-1]["market"] == "crypto"


def test_seeding_before_the_first_live_day_never_trades_before_it(monkeypatch):
    # Seeded while the day before the start was still an unfinished candle, then run the next day:
    # the first fill must be on the start date, decided at the close of the day before it.
    monkeypatch.setattr(live, "LIVE_START", "2026-10-01")
    full = make_panel(n_days=2650, freq="D")
    state = live.init_state(ACCTS)
    state, eq, trades, _ = live.run(
        state, {"crypto": full.truncate("2026-09-29")}, pd.Timestamp("2026-09-30"), MKT, ACCTS
    )
    assert eq == [] and trades == []
    state, eq, trades, _ = live.run(
        state, {"crypto": full.truncate("2026-10-02")}, pd.Timestamp("2026-10-03"), MKT, ACCTS
    )
    assert min(str(r["date"].date()) for r in eq) == "2026-10-01"
    assert min(str(t["date"].date()) for t in trades) == "2026-10-01"


def test_records_a_settings_row_only_when_the_choice_changes(monkeypatch):
    monkeypatch.setattr(live, "LIVE_START", "2026-10-01")
    full = make_panel(n_days=2650, freq="D")
    switch = pd.Timestamp("2026-10-03")
    real_targets = engines.targets

    def fake_targets(kind, market, panel, start):
        w = pd.DataFrame(0.0, index=panel.dates, columns=panel.tickers)
        if kind == "bench":
            return real_targets(kind, market, panel, start)
        chosen = pd.Series(["tour:a" if d < switch else "tour:b" for d in panel.dates], index=panel.dates)
        scores = pd.DataFrame({"tour:a": 0.0, "tour:b": 0.05}, index=panel.dates)
        return w, (lambda d, t, side: "hold cash"), {"chosen": chosen, "scores": scores}

    monkeypatch.setattr(live.engines, "targets", fake_targets)
    state = live.init_state(ACCTS)
    state, _, _, settings = live.run(
        state, {"crypto": full.truncate("2026-10-02")}, pd.Timestamp("2026-10-03"), MKT, ACCTS
    )
    assert [s["account"] for s in settings] == ["tournament_crypto"]
    assert settings[0]["variant"] == "tour:a" and "first run" in settings[0]["reason"]
    state, _, _, again = live.run(
        state, {"crypto": full.truncate("2026-10-02")}, pd.Timestamp("2026-10-03"), MKT, ACCTS
    )
    assert again == []
    state, _, _, later = live.run(
        state, {"crypto": full.truncate("2026-10-06")}, pd.Timestamp("2026-10-07"), MKT, ACCTS
    )
    assert len(later) == 1 and later[0]["variant"] == "tour:b" and str(later[0]["date"].date()) == "2026-10-03"
    assert "5.0%" in later[0]["reason"]
    assert state["accounts"]["tournament_crypto"]["variant"] == "tour:b"


def test_an_account_added_later_joins_at_its_first_run(monkeypatch):
    monkeypatch.setattr(live, "LIVE_START", "2026-10-01")
    p = make_panel(n_days=2650, freq="D").truncate("2026-10-05")
    state = live.init_state(ACCTS)
    state, *_ = live.run(state, {"crypto": p}, pd.Timestamp("2026-10-06"), MKT, ACCTS)
    more = {**ACCTS, "bench_eq_crypto": ("crypto", "bench_eq")}
    state, eq, trades, _ = live.run(state, {"crypto": p}, pd.Timestamp("2026-10-06"), MKT, more)
    assert "bench_eq_crypto" in state["accounts"]
    assert [r for r in eq if r["account"] == "bench_eq_crypto"]
    assert len({t["ticker"] for t in trades if t["account"] == "bench_eq_crypto"}) == len(MKT["crypto"].tickers)


def test_crypto_fee_matches_alpaca_tier_one():
    from paper_trader.config import CRYPTO, SLIPPAGE

    assert abs(CRYPTO.cost_rate - (0.0025 + SLIPPAGE)) < 1e-12
