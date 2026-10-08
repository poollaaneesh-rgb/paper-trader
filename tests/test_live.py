import copy

import pandas as pd
import pytest

from paper_trader import engines, live, selftune
from paper_trader.config import CRYPTO, FEE_CHANGE_NOTE, SLIPPAGE, Market
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


def test_a_new_benchmark_starts_on_the_live_start_date_and_leaves_the_other_records_alone(monkeypatch):
    # The equal-weight benchmarks begin on LIVE_START like bench_btc, not on the day they were added, and adding
    # them recomputes nothing for any account already running.
    monkeypatch.setattr(live, "LIVE_START", "2026-10-01")
    p = make_panel(n_days=2650, freq="D").truncate("2026-10-05")
    state = live.init_state(ACCTS)
    state, *_ = live.run(state, {"crypto": p}, pd.Timestamp("2026-10-06"), MKT, ACCTS)
    before = copy.deepcopy(state["accounts"])
    more = {**ACCTS, "bench_eq_crypto": ("crypto", "bench_eq")}
    state, eq, trades, _ = live.run(state, {"crypto": p}, pd.Timestamp("2026-10-06"), MKT, more)
    days = [str(r["date"].date()) for r in eq if r["account"] == "bench_eq_crypto"]
    assert days == ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05"]
    assert {str(t["date"].date()) for t in trades if t["account"] == "bench_eq_crypto"} == {"2026-10-01"}
    assert {r["account"] for r in eq} == {"bench_eq_crypto"} == {t["account"] for t in trades}
    assert all(state["accounts"][name] == before[name] for name in ACCTS)


def test_a_fee_change_applies_to_later_fills_only(monkeypatch):
    # A crypto fee change applies from the day it ships; fills already on record keep the rate they paid.
    monkeypatch.setattr(live, "LIVE_START", "2026-10-01")
    tickers = ("AAA", "BBB", "CCC", "DDD")
    old = {"crypto": Market("crypto", tickers, "AAA", 0.0055, 2)}
    new = {"crypto": Market("crypto", tickers, "AAA", 0.0030, 2)}
    flip = {"flip": ("crypto", "flip")}

    def flip_targets(kind, market, panel, start):  # holds AAA on even days and BBB on odd days: a fill every day
        w = pd.DataFrame(0.0, index=panel.dates, columns=panel.tickers)
        w.loc[panel.dates[panel.dates.day % 2 == 0], "AAA"] = 1.0
        w.loc[panel.dates[panel.dates.day % 2 == 1], "BBB"] = 1.0
        return w, (lambda d, t, side: "flip"), {}

    monkeypatch.setattr(live.engines, "targets", flip_targets)
    full = make_panel(n_days=2650, freq="D")
    state = live.init_state(flip)
    state, _, first, _ = live.run(state, {"crypto": full.truncate("2026-10-03")}, pd.Timestamp("2026-10-04"), old, flip)
    state, _, later, _ = live.run(state, {"crypto": full.truncate("2026-10-06")}, pd.Timestamp("2026-10-07"), new, flip)
    assert {str(t["date"].date()) for t in first} == {"2026-10-01", "2026-10-02", "2026-10-03"}
    assert {str(t["date"].date()) for t in later} == {"2026-10-04", "2026-10-05", "2026-10-06"}  # nothing re-emitted
    assert all(abs(t["cost"] / t["value"] - 0.0055) < 1e-9 for t in first)
    assert all(abs(t["cost"] / t["value"] - 0.0030) < 1e-9 for t in later)


def test_crypto_fee_matches_alpaca_tier_one():
    assert abs(CRYPTO.cost_rate - (0.0025 + SLIPPAGE)) < 1e-12


def _switch_reason(monkeypatch, market_key, lead):
    """Record variant `tour:a`, then let the layer move to `tour:b` with the given score lead; return that row."""
    monkeypatch.setattr(live, "LIVE_START", "2026-10-01")
    cost = 0.0055 if market_key == "crypto" else 0.0005
    markets = {market_key: Market(market_key, ("AAA", "BBB", "CCC", "DDD"), "AAA", cost, 2)}
    accts = {f"tournament_{market_key}": (market_key, "tournament")}
    switch = pd.Timestamp("2026-10-03")

    def fake_targets(kind, market, panel, start):
        w = pd.DataFrame(0.0, index=panel.dates, columns=panel.tickers)
        chosen = pd.Series(["tour:a" if d < switch else "tour:b" for d in panel.dates], index=panel.dates)
        scores = pd.DataFrame({"tour:a": 0.0, "tour:b": lead}, index=panel.dates)
        return w, (lambda d, t, side: "hold cash"), {"chosen": chosen, "scores": scores}

    monkeypatch.setattr(live.engines, "targets", fake_targets)
    full = make_panel(n_days=2650, freq="D")
    state = live.init_state(accts)
    state, _, _, first = live.run(
        state, {market_key: full.truncate("2026-10-02")}, pd.Timestamp("2026-10-03"), markets, accts
    )
    assert [r["variant"] for r in first] == ["tour:a"]
    state, _, _, later = live.run(
        state, {market_key: full.truncate("2026-10-06")}, pd.Timestamp("2026-10-07"), markets, accts
    )
    assert len(later) == 1 and later[0]["variant"] == "tour:b"
    return later[0]["reason"]


def test_the_fee_change_note_states_the_date_and_the_new_crypto_cost():
    assert "2026-10-07" in FEE_CHANGE_NOTE
    assert "0.25%" in FEE_CHANGE_NOTE and "0.05%" in FEE_CHANGE_NOTE
    assert "crypto" in FEE_CHANGE_NOTE and "re-scored" in FEE_CHANGE_NOTE


@pytest.mark.parametrize("lead", [0.005, -0.0002, float("nan")])
def test_a_crypto_setting_change_below_the_margin_cites_the_fee_change(monkeypatch, lead):
    reason = _switch_reason(monkeypatch, "crypto", lead)
    assert reason == FEE_CHANGE_NOTE
    assert "2026-10-07" in reason and "0.25%" in reason and "ahead of" not in reason


@pytest.mark.parametrize("lead", [selftune.MARGIN, 0.05])
def test_a_crypto_setting_change_at_or_above_the_margin_cites_the_score_lead(monkeypatch, lead):
    reason = _switch_reason(monkeypatch, "crypto", lead)
    assert (
        reason == f"ahead of the previous setting by {lead * 100:.1f}% over {selftune.WINDOW} trading days, after costs"
    )


def test_a_stocks_setting_change_below_the_margin_does_not_cite_the_crypto_fee(monkeypatch):
    reason = _switch_reason(monkeypatch, "stocks", -0.0002)
    assert reason == f"ahead of the previous setting by -0.0% over {selftune.WINDOW} trading days, after costs"


def test_an_account_trades_until_bankrupt_then_sells_out_and_keeps_its_record(monkeypatch):
    monkeypatch.setattr(live, "LIVE_START", "2026-10-01")
    full = make_panel(n_days=2650, freq="D").truncate("2026-10-08")
    crash = pd.Timestamp("2026-10-04")
    for frame in (full.open, full.close):
        frame.loc[crash:] *= 0.0001  # every coin loses 99.99% from the 4th
    real_targets = engines.targets

    def all_in(kind, market, panel, start):
        if kind == "bench":
            return real_targets(kind, market, panel, start)
        w = pd.DataFrame(0.0, index=panel.dates, columns=panel.tickers)
        w["AAA"] = 1.0
        return w, (lambda d, t, side: "all in on AAA"), {}

    monkeypatch.setattr(live.engines, "targets", all_in)
    state = live.init_state(ACCTS)
    state, eq, trades, _ = live.run(state, {"crypto": full}, pd.Timestamp("2026-10-09"), MKT, ACCTS)
    a = state["accounts"]["tournament_crypto"]
    assert a["bankrupt"] == "2026-10-04"
    assert a["positions"] == {} and a["pending"] is None
    sells = [t for t in trades if t["account"] == "tournament_crypto" and t["side"] == "sell"]
    assert [str(t["date"].date()) for t in sells] == ["2026-10-05"]
    assert sells[0]["reason"] == live.BANKRUPT_REASON
    after = [t for t in trades if t["account"] == "tournament_crypto" and t["date"] > pd.Timestamp("2026-10-05")]
    assert after == []
    days = [str(r["date"].date()) for r in eq if r["account"] == "tournament_crypto"]
    assert days[-1] == "2026-10-08"  # the equity record continues after the bankruptcy
    assert state["accounts"]["bench_btc"]["bankrupt"] is None  # a benchmark holds; it never goes bankrupt by trading
