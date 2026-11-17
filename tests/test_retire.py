"""Retired accounts (2026-11-16): never stepped again, record frozen at its last mark, still reported."""

import copy
import importlib.util
import json
from pathlib import Path

import pandas as pd

from paper_trader import challengers, config, engines, live, report
from paper_trader.config import Market
from tests.conftest import make_panel

ROOT = Path(__file__).resolve().parents[1]
CRYPTO_NAMES = ["tournament_crypto", "ml_crypto", "bench_btc", "bench_eq_crypto"]
STOCK_NAMES = ["tournament_stocks", "ml_stocks", "bench_spy", "bench_eq_stocks"]

MKTS = {
    "stocks": Market("stocks", ("AAA", "BBB", "CCC", "DDD"), "AAA", 0.0005, 2),
    "crypto": Market("crypto", ("AAA", "BBB", "CCC", "DDD"), "AAA", 0.003, 2),
}
ACCTS = {
    "tournament_stocks": ("stocks", "tournament"),
    "bench_spy": ("stocks", "bench"),
    "bench_eq_stocks": ("stocks", "bench_eq"),
    "tournament_crypto": ("crypto", "tournament"),
    "bench_btc": ("crypto", "bench"),
    "bench_eq_crypto": ("crypto", "bench_eq"),
}
RETIRE_ON = "2026-10-06"
CRYPTO_RETIRED = {
    n: {"date": RETIRE_ON, "cause": "test cause"} for n in ("tournament_crypto", "bench_btc", "bench_eq_crypto")
}


def _load_run_daily():
    spec = importlib.util.spec_from_file_location("run_daily", ROOT / "scripts" / "run_daily.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _panels(through):
    stocks = make_panel(n_days=1800, freq="B", seed=1).truncate(through)
    crypto = make_panel(n_days=2650, freq="D", seed=2).truncate(through)
    return {"stocks": stocks, "crypto": crypto}


def _fake_targets(monkeypatch, calls):
    """Cheap engines: the tournament holds AAA; benchmarks are the real ones. Records which markets were asked."""
    real = engines.targets

    def targets(kind, market, panel, start):
        calls.append(market.name)
        if kind in ("bench", "bench_eq"):
            return real(kind, market, panel, start)
        w = pd.DataFrame(0.0, index=panel.dates, columns=panel.tickers)
        w["AAA"] = 0.6
        w["BBB"] = 0.4
        return w, (lambda d, t, side: "fixed"), {}

    monkeypatch.setattr(live.engines, "targets", targets)


def test_the_four_crypto_accounts_are_retired_on_2026_11_16_quoting_the_user():
    assert sorted(config.RETIRED) == sorted(CRYPTO_NAMES)
    for name in CRYPTO_NAMES:
        assert config.RETIRED[name]["date"] == "2026-11-16"
        cause = config.RETIRED[name]["cause"]
        assert "retire all crypto bots, i do not want to invest in crypto anymore at all." in cause
    assert all(engines.ACCOUNTS[n][0] == "crypto" for n in config.RETIRED)
    assert all(n in engines.ACCOUNTS for n in STOCK_NAMES)  # nothing is deleted from the account list
    assert not set(STOCK_NAMES) & set(config.RETIRED)


def test_retirement_counts_from_its_date():
    table = {"x": {"date": "2026-11-16", "cause": "c"}}
    assert engines.retirement("x", pd.Timestamp("2026-11-15"), table) is None
    assert engines.retirement("x", pd.Timestamp("2026-11-16"), table) == table["x"]
    assert engines.retirement("y", pd.Timestamp("2026-11-16"), table) is None


def test_stock_results_are_byte_identical_with_crypto_retired(monkeypatch):
    monkeypatch.setattr(live, "LIVE_START", "2026-10-01")
    calls = []
    _fake_targets(monkeypatch, calls)
    stock_only = {k: v for k, v in ACCTS.items() if v[0] == "stocks"}

    def run_all(retired, panels_fn):
        monkeypatch.setattr(engines, "RETIRED", retired)
        state = live.init_state(ACCTS)
        out = []
        for through, today in (
            ("2026-10-04", "2026-10-05"),
            ("2026-10-05", "2026-10-06"),
            ("2026-10-09", "2026-10-10"),
            ("2026-10-16", "2026-10-17"),
        ):
            state, eq, tr, st = live.run(state, panels_fn(through), pd.Timestamp(today), MKTS, ACCTS)
            out.append(
                (
                    [r for r in eq if r["account"] in stock_only],
                    [r for r in tr if r["account"] in stock_only],
                    [r for r in st if r["account"] in stock_only],
                )
            )
        stocks_state = {n: state["accounts"][n] for n in stock_only}
        return (
            json.dumps(out, default=str, sort_keys=True),
            json.dumps(stocks_state, default=str, sort_keys=True),
            state,
        )

    base_rows, base_state, _ = run_all({}, _panels)
    calls.clear()

    # Retired from the second run on, and from then no crypto prices at all.
    def panels_without_crypto(through):
        p = _panels(through)
        if through != "2026-10-04":
            p["crypto"] = None
        return p

    rows, state_json, state = run_all(CRYPTO_RETIRED, panels_without_crypto)
    assert rows == base_rows
    assert '"account": "bench_spy"' in base_rows and '"account": "tournament_stocks"' in base_rows
    assert state_json == base_state
    assert state["accounts"]["bench_btc"]["last_date"] == "2026-10-04"  # stepped on the first run, then frozen
    assert not any(s["market"] == "crypto" for s in state["skips"])  # a retired market missing its prices is no skip


def test_a_retired_account_is_frozen_at_its_last_mark_and_never_stepped_again(monkeypatch):
    monkeypatch.setattr(live, "LIVE_START", "2026-10-01")
    calls = []
    _fake_targets(monkeypatch, calls)
    monkeypatch.setattr(engines, "RETIRED", CRYPTO_RETIRED)
    state = live.init_state(ACCTS)
    state, eq, _, _ = live.run(state, _panels("2026-10-04"), pd.Timestamp("2026-10-05"), MKTS, ACCTS)
    last = {r["account"]: r["equity"] for r in eq}
    frozen = copy.deepcopy({n: state["accounts"][n] for n in CRYPTO_RETIRED})
    calls.clear()
    state, eq, trades, settings = live.run(state, _panels("2026-10-20"), pd.Timestamp("2026-10-21"), MKTS, ACCTS)
    assert "crypto" not in calls  # no crypto engine (and no crypto model) is computed once retired
    assert not [r for r in eq + trades + settings if r["account"] in CRYPTO_RETIRED]
    assert {n: state["accounts"][n] for n in CRYPTO_RETIRED} == frozen  # holdings, cash and last day unchanged
    assert frozen["tournament_crypto"]["last_date"] == "2026-10-04"
    assert last["bench_btc"] > 0
    assert [r for r in eq if r["account"] == "bench_spy"]  # stocks carry on


def test_a_retired_account_ends_its_challengers_and_refuses_new_ones(monkeypatch):
    monkeypatch.setattr(engines, "RETIRED", {"tournament_crypto": {"date": "2026-10-06", "cause": "c"}})
    state = live.init_state({"tournament_crypto": ("crypto", "tournament")})
    state["challengers"] = {
        "old": {"account": "tournament_crypto", "status": "active", "variant": {}, "started": "2026-10-02"}
    }
    entry = {
        "id": "new-one",
        "account": "tournament_crypto",
        "variant": {"strategies": ["momentum"]},
        "hypothesis": "h",
        "rule_id": "R1",
        "started": "2026-10-07",
        "status": "active",
    }
    book = challengers.Book([entry])
    accts = {"tournament_crypto": ("crypto", "tournament")}
    live.run(state, {"crypto": None}, pd.Timestamp("2026-10-07"), {"crypto": MKTS["crypto"]}, accts, book)
    assert state["challengers"]["old"]["status"] == "retired"
    assert "retired" in state["challengers"]["old"]["result"]
    assert "new-one" not in state["challengers"]
    assert "retired" in state["challengers_refused"]["new-one"]["reason"]


def test_the_summary_and_feed_mark_retired_accounts_and_keep_their_final_values(tmp_path, monkeypatch):
    rd = _load_run_daily()
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    monkeypatch.setattr(rd, "LIVE", live_dir)
    monkeypatch.setattr(engines, "RETIRED", {"bench_btc": {"date": "2026-11-16", "cause": "his words"}})
    rows = [
        ("2026-10-01", "bench_btc", 400.0),
        ("2026-11-14", "bench_btc", 320.0),
        ("2026-11-15", "bench_btc", 310.68),
        ("2026-10-01", "bench_spy", 400.0),
        ("2026-11-14", "bench_spy", 420.0),
    ]
    pd.DataFrame(rows, columns=["date", "account", "equity"]).to_csv(live_dir / "equity.csv", index=False)
    states = {"bench_btc": {"pending": {"BTC-USD": 1.0}, "bankrupt": None}}
    out = rd.live_summary([], states, today=pd.Timestamp("2026-11-17"))
    btc, spy = out["accounts"]["bench_btc"], out["accounts"]["bench_spy"]
    assert btc["retired"] == "2026-11-16" and btc["retired_cause"] == "his words"
    assert btc["final_equity"] == 310.68 and btc["last_date"] == "2026-11-15"
    assert btc["targets"] == {} and btc["return_30d"] is None  # a retired account can't hold or be copied by a seat
    assert spy["retired"] is None and spy["retired_cause"] is None
    before = rd.live_summary([], states, today=pd.Timestamp("2026-11-15"))["accounts"]["bench_btc"]
    assert before["retired"] is None and before["targets"] == {"BTC-USD": 1.0}

    summary = {
        "generated_at": "2026-11-17T04:30+00:00",
        "live": {"start": "2026-10-01", "accounts": out["accounts"]},
        "backtest": {"accounts": {}},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary, default=str))
    report.build(tmp_path, tmp_path / "site")
    feed = json.loads((tmp_path / "site" / "summary.json").read_text())
    assert feed["live"]["bench_btc"]["retired"] == "2026-11-16"
    assert feed["live"]["bench_btc"]["retired_cause"] == "his words"
    assert feed["live"]["bench_btc"]["final_equity"] == 310.68
    assert feed["live"]["bench_btc"]["last_date"] == "2026-11-15"
    assert "retired" in (tmp_path / "site" / "index.html").read_text()


def test_the_nightly_run_fetches_no_prices_for_a_market_whose_accounts_are_all_retired(monkeypatch):
    rd = _load_run_daily()
    monkeypatch.setattr(engines, "RETIRED", config.RETIRED)
    assert rd.markets_to_fetch(pd.Timestamp("2026-11-17")) == ["stocks"]
    assert rd.markets_to_fetch(pd.Timestamp("2026-11-15")) == ["stocks", "crypto"]
