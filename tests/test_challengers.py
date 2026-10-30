import copy
import json
import math

import numpy as np
import pandas as pd
import pytest

from paper_trader import challengers, engines, live, variants
from paper_trader.config import CRYPTO, Market
from paper_trader.data import Panel

MKT = {"crypto": Market("crypto", ("AAA", "BBB"), "AAA", 0.0, 2)}
ACCTS = {"tournament_crypto": ("crypto", "tournament")}
TOUR = {"speeds": "fast", "lookback": 21, "cadence": "W"}


def _panel(drift: float) -> Panel:
    """AAA moves by `drift` a day, opening where it closes; BBB stays at 100."""
    dates = pd.date_range("2026-01-01", "2027-06-30", freq="D")
    close = pd.DataFrame({"AAA": 100 * (1 + drift) ** np.arange(len(dates)), "BBB": 100.0}, index=dates)
    return Panel(open=close.copy(), close=close, volume=close * 0 + 1)


def _entry(cid="hold-aaa", started="2026-10-10", **changes):
    entry = {
        "id": cid,
        "account": "tournament_crypto",
        "variant": dict(TOUR),
        "hypothesis": "holding the leader beats sitting in cash",
        "rule_id": "R1",
        "started": started,
        "status": "active",
    }
    entry.update(changes)
    return entry


def _run(state, panel, through, entries=()):
    """One nightly run that has seen every complete day through `through`."""
    book = challengers.Book(list(entries))
    today = pd.Timestamp(through) + pd.Timedelta(days=1)
    state, _, trades, settings = live.run(state, {"crypto": panel.truncate(through)}, today, MKT, ACCTS, book)
    return state, trades, settings, book


@pytest.fixture
def fakes(monkeypatch):
    """The live account sits in cash; every challenger puts everything in AAA."""
    monkeypatch.setattr(live, "LIVE_START", "2026-10-01")

    def cash(kind, market, panel, start):
        return pd.DataFrame(0.0, index=panel.dates, columns=panel.tickers), (lambda d, t, side: "cash"), {}

    def all_in(kind, market, panel, definition, diag):
        w = pd.DataFrame(0.0, index=panel.dates, columns=panel.tickers)
        w["AAA"] = 1.0
        return w, (lambda d, t, side: "all in AAA")

    monkeypatch.setattr(live.engines, "targets", cash)
    monkeypatch.setattr(live.engines, "challenger_targets", all_in)


def test_a_challenger_added_today_has_no_score_today_and_no_backfilled_history(fakes):
    panel = _panel(0.002)
    state, *_ = _run(live.init_state(ACCTS), panel, "2026-10-10")  # the live record runs through Oct 10
    entries = [_entry(started="2026-10-11")]  # the review adds it on Oct 11
    state, _, _, book = _run(state, panel, "2026-10-10", entries)  # Oct 11's run: no new day yet
    assert book.rows == [] and "hold-aaa" not in state["challengers"]
    state, _, _, book = _run(state, panel, "2026-10-11", entries)  # Oct 11 closes: the day it was added
    st = state["challengers"]["hold-aaa"]
    assert book.rows == [] and st["days"] == 0 and st["lead"] is None and st["equity"] is None
    assert st["last_date"] == "2026-10-11"
    state, _, _, book = _run(state, panel, "2026-10-14", entries)
    assert [str(r["date"].date()) for r in book.rows] == ["2026-10-12", "2026-10-13", "2026-10-14"]
    assert book.rows[0]["equity"] == 400.0  # it starts from a copy of the live book, which was all cash
    assert state["challengers"]["hold-aaa"]["days"] == 3
    snapshot = copy.deepcopy(state)
    state, _, _, again = _run(state, panel, "2026-10-14", entries)
    assert again.rows == [] and state == snapshot


def test_a_backdated_challenger_is_refused_and_never_computed(fakes):
    panel = _panel(0.002)
    state, *_ = _run(live.init_state(ACCTS), panel, "2026-10-10")
    state, _, _, book = _run(state, panel, "2026-10-14", [_entry(started="2026-10-05")])
    assert "hold-aaa" not in state["challengers"] and book.rows == []
    reason = state["challengers_refused"]["hold-aaa"]["reason"]
    assert "before tournament_crypto's last recorded day 2026-10-10" in reason and "backfilled" in reason


@pytest.mark.parametrize(
    "changes, key",
    [
        ({"variant": {**TOUR, "fee": 0.001}}, "fee"),
        ({"start_cash": 1000}, "start_cash"),
        ({"variant": {**TOUR, "universe": ["AAA"]}}, "universe"),
        ({"variant": {**TOUR, "slippage": 0.0}}, "slippage"),
        ({"variant": {**TOUR, "max_positions": 1}}, "max_positions"),
        ({"variant": {**TOUR, "params": {"momentum": {"cost_rate": 0}}}}, "cost_rate"),
    ],
)
def test_an_entry_that_sets_risk_or_capital_is_refused_with_its_reason(fakes, changes, key):
    panel = _panel(0.002)
    state, *_ = _run(live.init_state(ACCTS), panel, "2026-10-10")
    state, _, _, book = _run(state, panel, "2026-10-14", [_entry(**changes)])
    assert state["challengers"] == {} and book.rows == []
    reason = state["challengers_refused"]["hold-aaa"]["reason"]
    assert reason.startswith("sets risk or capital parameters") and key in reason
    assert any(line.startswith("challenger hold-aaa refused: sets risk") for line in book.events)


def test_settings_outside_the_menus_parameters_are_refused():
    assert (
        variants.check_definition("tournament", {**TOUR, "stop_loss": 0.1}) == "the tournament has no setting stop_loss"
    )
    assert variants.check_definition("tournament", {**TOUR, "params": {"trend": {"width": 3}}}) == (
        "trend has no setting width"
    )
    assert variants.check_definition("tournament", {"lookback": 21}) == "cadence must be one of M, W, D"
    assert variants.check_definition("ml", {"entry": 0.57, "gap": 0.05}) == "the ML band rule needs hold"
    assert variants.check_definition("ml", {"entry": 0.4, "gap": 0.0, "hold": 1}).startswith("entry must be")
    assert variants.check_definition("bench", {}) == "the bench engine has no settings to change"
    assert variants.check_definition("ml", {"entry": 0.57, "gap": 0.05, "hold": 3}) is None
    assert variants.check_definition("tournament", {**TOUR, "params": {"momentum": {"lookback": 84}}}) is None


def test_the_bar_needs_thirty_days_and_a_two_percent_lead():
    assert challengers.verdict(29, 0.50) is None
    assert challengers.verdict(30, 0.0199) is None
    assert challengers.verdict(30, challengers.MARGIN) == "promote"
    assert challengers.verdict(30, -challengers.MARGIN) == "retire"
    assert challengers.verdict(119, 0.01) is None
    assert challengers.verdict(120, 0.01) == "retire"
    assert challengers.verdict(120, 0.02) == "promote"


def test_the_account_switches_only_once_the_challenger_clears_the_bar(fakes):
    # AAA rises 0.2% a day while the live account sits in cash: the challenger leads by 2% from its 12th day, but the
    # bar also needs 30 days of its own, so it takes over at the close of its 30th day (Oct 11 + 29 = Nov 9).
    panel = _panel(0.002)
    entries = [_entry(started="2026-10-10")]
    state, *_ = _run(live.init_state(ACCTS), panel, "2026-10-09")
    state, _, settings, book = _run(state, panel, "2026-11-08", entries)
    st = state["challengers"]["hold-aaa"]
    assert settings == [] and st["status"] == "active" and st["days"] == 29 and st["lead"] > challengers.MARGIN
    assert book.rows[11]["equity"] / 400 - 1 > 0.02  # the lead was there from day 12
    state, trades, settings, book = _run(state, panel, "2026-11-30", entries)
    assert len(settings) == 1
    row = settings[0]
    assert str(row["date"].date()) == "2026-11-09" and row["variant"] == "review:hold-aaa"
    assert "review challenger hold-aaa (rule R1)" in row["reason"]
    assert "over its 30 trading days since 2026-10-10" in row["reason"]
    st = state["challengers"]["hold-aaa"]
    assert st["status"] == "promoted" and st["ended"] == "2026-11-09" and st["days"] == 30
    assert [str(r["date"].date()) for r in book.rows] == ["2026-11-09"]  # its own record stops when it takes over
    a = state["accounts"]["tournament_crypto"]
    assert a["following"]["id"] == "hold-aaa" and a["variant"] == "review:hold-aaa"
    first = [t for t in trades if t["account"] == "tournament_crypto"]
    assert str(first[0]["date"].date()) == "2026-11-10" and first[0]["ticker"] == "AAA" and first[0]["side"] == "buy"
    assert book.sync(state) and entries[0]["status"] == "promoted" and entries[0]["ended"] == "2026-11-09"


def test_one_run_or_many_give_the_same_challenger_record(fakes):
    panel = _panel(0.002)
    entries = [_entry(started="2026-10-10")]
    base, *_ = _run(live.init_state(ACCTS), panel, "2026-10-10")
    once, _, _, book = _run(copy.deepcopy(base), panel, "2026-11-20", entries)
    nightly, rows = copy.deepcopy(base), []
    for d in pd.date_range("2026-10-11", "2026-11-20"):
        nightly, _, _, b = _run(nightly, panel, d, entries)
        rows += b.rows
    assert rows == book.rows and nightly["challengers"] == once["challengers"]


@pytest.mark.parametrize(
    "drift, days, result",
    [(-0.002, 30, "behind the live setting by"), (0.0, 120, "short of the bar")],
)
def test_a_challenger_is_retired_when_it_trails_after_30_days_or_has_not_cleared_by_120(fakes, drift, days, result):
    panel = _panel(drift)
    state, *_ = _run(live.init_state(ACCTS), panel, "2026-10-10")
    state, _, settings, book = _run(state, panel, "2027-06-29", [_entry(started="2026-10-10")])
    st = state["challengers"]["hold-aaa"]
    assert settings == [] and st["status"] == "retired" and result in st["result"]
    assert st["days"] == days and len(book.rows) == days
    assert st["ended"] == str((pd.Timestamp("2026-10-10") + pd.Timedelta(days=days)).date())
    assert "following" not in state["accounts"]["tournament_crypto"]


def test_an_account_takes_at_most_two_active_challengers(fakes):
    panel = _panel(0.002)
    state, *_ = _run(live.init_state(ACCTS), panel, "2026-10-10")
    entries = [_entry(cid=f"c{i}") for i in range(3)]
    state, _, _, book = _run(state, panel, "2026-10-12", entries)
    assert sorted(state["challengers"]) == ["c0", "c1"]
    assert state["challengers_refused"]["c2"]["reason"] == "tournament_crypto already has 2 active challengers"
    assert {r["challenger"] for r in book.rows} == {"c0", "c1"}


def test_a_started_challenger_keeps_its_setting_and_can_only_be_withdrawn(fakes):
    panel = _panel(0.002)
    entries = [_entry()]
    state, *_ = _run(live.init_state(ACCTS), panel, "2026-10-10")
    state, *_ = _run(state, panel, "2026-10-12", entries)
    entries[0]["variant"] = {**TOUR, "lookback": 63}
    state, _, _, book = _run(state, panel, "2026-10-13", entries)
    assert state["challengers"]["hold-aaa"]["variant"] == TOUR and len(book.rows) == 1
    assert "keeps running" in state["challengers_refused"]["hold-aaa"]["reason"]
    entries[0]["variant"] = dict(TOUR)  # put back: the note goes
    state, *_ = _run(state, panel, "2026-10-13", entries)
    assert "hold-aaa" not in state["challengers_refused"]
    entries[0]["status"] = "retired"
    state, _, _, book = _run(state, panel, "2026-10-14", entries)
    assert state["challengers"]["hold-aaa"]["result"] == "withdrawn by the review" and book.rows == []


def test_written_settings_match_the_menu_entries_with_the_same_numbers(panel):
    rng = np.random.default_rng(1)
    probs = pd.DataFrame(rng.uniform(0.4, 0.7, (60, 4)), index=panel.dates[:60], columns=panel.tickers)
    ml = variants.ml_variants(probs, 2)
    pd.testing.assert_frame_equal(
        variants.ml_definition(probs, 2, {"entry": 0.55, "gap": 0.03, "hold": 5}), ml["ml:in55-out52-hold5"]
    )
    pd.testing.assert_frame_equal(
        variants.ml_definition(probs, 2, {"rule": "rerank", "threshold": 0.55}), ml["ml:rerank55"]
    )
    pd.testing.assert_frame_equal(variants.ml_definition(probs, 2, {"rule": "cash"}), ml["ml:cash"])
    tour = variants.tournament_variants(panel, 2, 0.005)
    w, _ = variants.tournament_definition(panel, 2, 0.005, {"speeds": "slow", "lookback": 63, "cadence": "W"})
    pd.testing.assert_frame_equal(w, tour["tour:slow-63-W-net"])
    fixed, _ = variants.tournament_definition(panel, 2, 0.005, {"lookback": 63, "cadence": "M", "scoring": "gross"})
    pd.testing.assert_frame_equal(fixed, tour[variants.TOUR_FIXED])
    assert variants.describe_definition("tournament", {"speeds": "slow", "lookback": 126, "cadence": "W"}) == (
        variants.describe("tour:slow-126-W-net")
    )
    assert variants.describe_definition("ml", {"entry": 0.58, "gap": 0.07, "hold": 10}) == (
        variants.describe("ml:in58-out51-hold10")
    )


def test_a_crypto_challenger_is_gated_like_its_account():
    dates = pd.date_range("2025-01-01", periods=300, freq="D")
    btc = pd.Series(100 * 0.995 ** np.arange(300), index=dates)  # a steady fall: the gate stays shut
    close = pd.DataFrame({"BTC-USD": btc, "ETH-USD": 50.0}, index=dates)
    panel = Panel(open=close, close=close, volume=close * 0 + 1)
    probs = pd.DataFrame(0.9, index=dates, columns=panel.tickers)
    definition = {"entry": 0.55, "gap": 0.0, "hold": 1}
    assert (variants.ml_definition(probs, 2, definition).sum(axis=1) > 0).all()
    w, explain = engines.challenger_targets("ml", CRYPTO, panel, definition, {"probs": probs})
    assert (w == 0).all().all()
    assert "waits in cash" in explain(dates[-1], "ETH-USD", "sell")


def test_the_check_command_reports_each_entry(tmp_path):
    live_dir = tmp_path / "results" / "live"
    live_dir.mkdir(parents=True)
    state = live.init_state()
    for a in state["accounts"].values():
        a.update(initialised=True, last_date="2026-10-28")
    (live_dir / "state.json").write_text(json.dumps(state))
    good = {**_entry(cid="ok", started="2026-10-29"), "account": "ml_stocks"}
    good["variant"] = {"entry": 0.57, "gap": 0.05, "hold": 3}
    (live_dir / "challengers.json").write_text(json.dumps([good]))
    assert challengers.main(tmp_path) == 0
    (live_dir / "challengers.json").write_text(json.dumps([good, _entry(cid="old", started="2026-10-01")]))
    assert challengers.main(tmp_path) == 1


def test_the_lead_is_the_log_return_gap_over_the_same_days():
    assert challengers._lead(408.0, 400.0) == pytest.approx(math.log(1.02))
