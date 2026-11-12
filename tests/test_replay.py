import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from paper_trader import data, decisions, live, replay
from paper_trader.config import CRYPTO
from tests.conftest import make_panel

ROOT = Path(__file__).resolve().parents[1]
ACCTS = {"tournament_crypto": ("crypto", "tournament"), "bench_btc": ("crypto", "bench")}


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture(scope="module")
def night(tmp_path_factory):
    """A repo with this code, a frozen crypto panel and the decision records of a live run through Oct 5."""
    repo = tmp_path_factory.mktemp("repo")
    shutil.copytree(ROOT / "paper_trader", repo / "paper_trader", ignore=shutil.ignore_patterns("__pycache__"))
    _git(repo, "init", "-q")
    _git(repo, "add", "paper_trader")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "code")
    code = decisions.code_version(repo)
    out = repo / "results" / "live"
    out.mkdir(parents=True)
    fresh = make_panel(n_days=2650, tickers=CRYPTO.tickers, freq="D").truncate("2026-10-05")
    data.save_frozen(out / "prices_crypto.csv", fresh, {})
    panel = data._scaled(*data.load_frozen(out / "prices_crypto.csv"))  # as run_daily trades on it
    rows = []
    state = live.init_state(ACCTS)
    state, *_ = live.run(state, {"crypto": panel}, pd.Timestamp("2026-10-06"), {"crypto": CRYPTO}, ACCTS, None, rows)
    for r in rows:
        r["code"] = code
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df.to_csv(out / "decisions.csv", index=False)
    return repo, rows, state, panel


def test_each_decision_records_its_data_date_digest_setting_and_targets(night):
    _, rows, state, panel = night
    tour = [r for r in rows if r["account"] == "tournament_crypto"]
    days = [str(r["date"].date()) for r in tour]
    assert days == ["2026-09-30", "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05"]
    assert all(r["data_date"] == "2026-10-05" and r["data_digest"] == decisions.digest(panel) for r in rows)
    assert decisions.parse_targets(tour[-1]["targets"]) == pytest.approx(
        {t: w for t, w in state["accounts"]["tournament_crypto"]["pending"].items() if w}, abs=1e-6
    )
    assert tour[-1]["setting"] == state["accounts"]["tournament_crypto"]["variant"]


def test_targets_text_round_trips_hold_cash_and_weights():
    assert decisions.parse_targets(decisions.targets_text(None)) is None
    assert decisions.parse_targets(decisions.targets_text({"A": 0.0})) == {}
    assert decisions.parse_targets(decisions.targets_text({"B": 0.25, "A": 0.5})) == {"A": 0.5, "B": 0.25}


def test_a_recorded_day_reproduces_from_its_code_version_in_a_sandbox(night):
    repo, *_ = night
    before = {p: p.read_bytes() for p in (repo / "results" / "live").iterdir()}
    res = replay.reproduce(repo, "tournament_crypto", "2026-10-03")
    assert res["reproduced"] is True, res
    assert {p: p.read_bytes() for p in (repo / "results" / "live").iterdir()} == before  # nothing written


def test_a_record_the_code_does_not_reproduce_is_reported(night, tmp_path):
    repo, *_ = night
    path = repo / "results" / "live" / "decisions.csv"
    original = path.read_text()
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df.loc[(df.account == "tournament_crypto") & (df.date == "2026-10-03"), "targets"] = "BTC-USD=1.000000"
        df.to_csv(path, index=False)
        assert replay.reproduce(repo, "tournament_crypto", "2026-10-03")["reproduced"] is False
    finally:
        path.write_text(original)


def test_a_changed_panel_is_caught_by_the_digest(night):
    repo, *_ = night
    panel, source = replay.rebuild_panel(repo, "crypto", "2026-10-05", "000000000000")
    assert panel is None and "matches digest" in source


def test_a_fix_is_scored_on_every_day_but_the_one_that_prompted_it(night):
    repo, _, _, panel = night
    fix = {"speeds": "fast", "lookback": 21, "cadence": "D"}
    out = replay.try_fix(repo, "tournament_crypto", "2026-10-03", fix, panel=panel)
    # Decisions Sep 30 .. Oct 5; Oct 4 and 5 have no second open yet, Oct 3 is the trigger: Sep 30, Oct 1, Oct 2.
    assert out["days"] == 3 and out["unrecorded_days"] == 0
    assert out["passes"] == (out["changes_day"] and out["lead"] > replay.BAR)


def test_decision_returns_hold_until_the_next_open_and_pay_for_changes():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    opens = pd.DataFrame({"A": [100.0, 100.0, 110.0, 110.0]}, index=dates)
    p = data.Panel(opens, opens, opens)
    r = replay._decision_returns(p, {dates[0]: {"A": 1.0}, dates[1]: None}, 0.01)
    assert r[dates[0]] == pytest.approx(0.10 - 0.01)  # bought at the open of day 1, held to the open of day 2
    assert r[dates[1]] == pytest.approx(0.0)  # hold: same targets, no cost, flat move


def test_the_bar_is_a_lead_of_more_than_a_quarter_of_lives_result():
    assert replay.per_dollar_lead(0.125, 0.10) == pytest.approx(0.25)
    assert replay.per_dollar_lead(-0.05, -0.10) == pytest.approx(0.5)
    assert json.dumps(replay.BAR) == "0.25"


def test_each_decision_carries_the_shared_record_fields(night):
    """Step 4 of the revamp: id, team, book, where the signal came from, the names passed over, and the limits."""
    _, rows, _, _ = night
    assert all(set(decisions.FIELDS) <= set(r) for r in rows)
    assert len({r["id"] for r in rows}) == len(rows)
    tour = [r for r in rows if r["account"] == "tournament_crypto"][-1]
    assert tour["id"] == "tournament_crypto|2026-10-05"
    assert (tour["team"], tour["book"], tour["signal_source"], tour["model_id"]) == ("crypto", "rules", "rule", "")
    limits = json.loads(tour["limits"])
    assert limits["cost_rate"] == CRYPTO.cost_rate and limits["max_positions"] == CRYPTO.max_positions
    # Every coin with a close that night is a candidate, held or not, so a name the account passed over is on record.
    names = [part.split("=")[0] for part in tour["candidates"].split(";")]
    assert names == sorted(CRYPTO.tickers)
    assert len(decisions.parse_targets(tour["targets"]) or {}) <= len(names)
    bench = [r for r in rows if r["account"] == "bench_btc"][-1]
    assert bench["signal_source"] == "rule" and bench["model_id"] == ""


def test_a_model_account_records_its_odds_for_every_name_scored(night):
    _, _, _, panel = night
    day = panel.dates[-1]
    probs = pd.DataFrame(float("nan"), index=panel.dates, columns=panel.tickers)
    probs.loc[day, panel.tickers[0]] = 0.6123
    probs.loc[day, panel.tickers[1]] = 0.4
    a = {"variant": "ml:in55-out55-hold5", "pending": {panel.tickers[0]: 1.0}}
    r = decisions.row(day, "ml_crypto", panel, "abc", a, "ml", CRYPTO, probs)
    assert (r["team"], r["signal_source"], r["model_id"]) == ("crypto", "model", "ml:in55-out55-hold5")
    got = dict(part.split("=") for part in r["candidates"].split(";"))
    assert got[panel.tickers[0]] == "0.6123" and got[panel.tickers[1]] == "0.4000"
    assert sorted(got) == sorted(panel.tickers) and sum(1 for v in got.values() if v) == 2  # unscored names stay listed
    # The same account following a promoted challenger names that challenger's setting as its model.
    a["following"] = {"variant": {"kind": "ml", "enter": 0.6}}
    assert json.loads(decisions.row(day, "ml_crypto", panel, "abc", a, "ml", CRYPTO, probs)["model_id"]) == {
        "enter": 0.6,
        "kind": "ml",
    }


def test_widen_adds_blank_columns_and_changes_no_old_value(tmp_path):
    path = tmp_path / "decisions.csv"
    old = pd.DataFrame(
        [
            {
                "date": "2026-11-05",
                "account": "ml_stocks",
                "data_date": "2026-11-05",
                "data_digest": "b21fd60581df",
                "code": "437aa92018",
                "setting": "ml:in55-out55-hold5",
                "following": "",
                "targets": "AAPL=0.200000;XOM=0.200000",
            }
        ]
    )
    old.to_csv(path, index=False)
    before = path.read_text().splitlines()
    assert decisions.widen(path) is True
    df = decisions.load(path)
    assert list(df.columns) == list(decisions.FIELDS)
    assert df[list(decisions.BASE_FIELDS)].to_dict("records") == old.to_dict("records")
    assert all(df[c].iloc[0] == "" for c in decisions.RECORD_FIELDS)
    assert path.read_text().splitlines()[1].startswith(before[1])
    assert decisions.widen(path) is False  # a second night changes nothing
    # A new, wider row appended under the widened header reads back whole.
    new = {c: "" for c in decisions.FIELDS} | {
        "date": "2026-11-12",
        "account": "ml_stocks",
        "id": "ml_stocks|2026-11-12",
    }
    pd.DataFrame([new]).to_csv(path, mode="a", header=False, index=False)
    assert decisions.load(path)["id"].tolist() == ["", "ml_stocks|2026-11-12"]
