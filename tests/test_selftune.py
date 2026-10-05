import numpy as np
import pandas as pd

from paper_trader import selftune
from paper_trader.data import Panel

DATES = pd.date_range("2024-01-01", periods=300, freq="B")


def _returns(a, b):
    return pd.DataFrame({"a": a, "b": b}, index=DATES)


def test_choose_returns_the_default_without_enough_history():
    rets = _returns(np.zeros(300), np.full(300, 0.001))
    chosen = selftune.choose(rets.iloc[:50], "a", window=126, margin=0.02)
    assert (chosen == "a").all()


def test_choose_switches_only_once_the_lead_passes_the_margin():
    b = np.full(300, 0.0005)
    b[100:200] = 0.002  # b pulls ahead from day 100
    b[200:] = -0.002  # then falls behind
    chosen = selftune.choose(_returns(np.zeros(300), b), "a", window=20, margin=0.02)
    assert (chosen.iloc[:106] == "a").all()  # 0.0005 a day over 20 days is only 1%: not enough
    assert chosen.iloc[106] == "b"  # 7 days at 0.002 plus 13 at 0.0005 is the first 20-day lead over 2%
    assert (chosen.iloc[106:214] == "b").all()  # switching back needs a to lead by the margin too
    assert chosen.iloc[214] == "a"


def test_choose_has_no_lookahead():
    rng = np.random.default_rng(3)
    rets = _returns(rng.normal(0, 0.01, 300), rng.normal(0.0005, 0.01, 300))
    full = selftune.choose(rets, "a", window=30, margin=0.01)
    for n in (120, 200, 299):
        part = selftune.choose(rets.iloc[:n], "a", window=30, margin=0.01)
        pd.testing.assert_series_equal(full.iloc[:n], part, check_names=False)


def test_combine_takes_the_chosen_variants_row():
    idx = DATES[:4]
    va = pd.DataFrame({"X": [1.0, 1.0, 1.0, 1.0], "Y": [0.0] * 4}, index=idx)
    vb = pd.DataFrame({"X": [0.0] * 4, "Y": [0.5] * 4}, index=idx)
    chosen = pd.Series(["a", "a", "b", "a"], index=idx)
    w = selftune.combine({"a": va, "b": vb}, chosen)
    assert w.loc[idx[1]].tolist() == [1.0, 0.0]
    assert w.loc[idx[2]].tolist() == [0.0, 0.5]


def test_net_returns_match_a_hand_computed_case():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    opens = pd.DataFrame({"AAA": [100.0, 101.0, 103.0, 103.0, 105.0]}, index=idx)
    panel = Panel(open=opens, close=opens * 1.001, volume=opens * 0 + 1)
    w = pd.DataFrame({"AAA": [1.0] * 5}, index=idx)
    r = selftune.net_returns(w, panel, 0.001)
    # Weights decided at close t earn open t+1 -> open t+2; the first row's turnover (0 -> 100%) is charged with it.
    assert np.allclose(r.to_numpy(), [0.0, 0.0, 103 / 101 - 1 - 0.001, 0.0, 105 / 103 - 1])


def test_run_scores_and_choice_line_up():
    rng = np.random.default_rng(5)
    idx = DATES
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (300, 2)), axis=0)), index=idx, columns=["X", "Y"])
    panel = Panel(open=close.shift(1).bfill(), close=close, volume=close * 0 + 1)
    only_x = pd.DataFrame({"X": 1.0, "Y": 0.0}, index=idx)
    only_y = pd.DataFrame({"X": 0.0, "Y": 1.0}, index=idx)
    w, chosen, scores = selftune.run({"x": only_x, "y": only_y}, panel, 0.0, "x", window=40, margin=0.0)
    assert set(chosen.unique()) <= {"x", "y"}
    assert list(scores.columns) == ["x", "y"]
    # Whenever the choice changes to y, y's score was ahead of x's that day.
    switched = chosen[(chosen == "y") & (chosen.shift(1) == "x")].index
    assert len(switched) > 0
    assert (scores.loc[switched, "y"] > scores.loc[switched, "x"]).all()
    assert (w.loc[chosen == "y", "Y"] == 1.0).all()
