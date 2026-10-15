import numpy as np
import pandas as pd

from paper_trader import data, live
from paper_trader.data import Panel
from tests.conftest import make_panel


def _noisy(p: Panel, seed=1) -> Panel:
    rng = np.random.default_rng(seed)
    f = 1 + rng.normal(0, 1e-6, size=p.close.shape)
    return Panel(p.open * f, p.close * f, p.volume)


def test_first_run_stores_the_download():
    fresh = make_panel(50)
    panel, raw, scale, splits = data.freeze(fresh, None)
    assert panel is fresh and raw is fresh and scale == {} and splits == {}


def test_days_already_used_stay_as_first_seen_and_new_days_are_added():
    full = make_panel(60)
    _, raw, scale, _ = data.freeze(full.truncate(full.dates[49]), None)
    panel, raw2, scale2, splits = data.freeze(_noisy(full), raw, scale)
    assert len(panel.dates) == 60 and scale2 == {} and splits == {}
    pd.testing.assert_frame_equal(panel.close.iloc[:50], full.close.iloc[:50])  # the download noise is ignored
    pd.testing.assert_frame_equal(raw2.close.iloc[:50], raw.close)  # the stored rows never change


def test_a_dividend_rebase_scales_the_ticker_without_calling_it_a_split():
    full = make_panel(60)
    _, raw, scale, _ = data.freeze(full.truncate(full.dates[49]), None)
    rebased = Panel(full.open.copy(), full.close.copy(), full.volume)
    rebased.close["AAA"] *= 0.995
    rebased.open["AAA"] *= 0.995
    panel, _, scale, splits = data.freeze(rebased, raw, scale)
    assert splits == {} and abs(scale["AAA"]["price"] - 0.995) < 1e-9
    assert np.allclose(panel.close["AAA"].iloc[:50], full.close["AAA"].iloc[:50] * 0.995)


def test_a_split_rescales_the_history_and_the_accounts_units(tmp_path):
    full = make_panel(60)
    _, raw, scale, _ = data.freeze(full.truncate(full.dates[49]), None)
    split = Panel(full.open.copy(), full.close.copy(), full.volume.copy())
    split.close["AAA"] /= 10
    split.open["AAA"] /= 10
    split.volume["AAA"] *= 10
    panel, raw2, scale, splits = data.freeze(split, raw, scale)
    assert abs(splits["AAA"] - 0.1) < 1e-9
    assert np.allclose(panel.close["AAA"], split.close["AAA"])
    data.save_frozen(tmp_path / "prices_stocks.csv", raw2, scale)
    raw3, scale3 = data.load_frozen(tmp_path / "prices_stocks.csv")
    assert scale3 == scale and np.allclose(raw3.close, raw2.close, equal_nan=True)
    state = {"accounts": {"x": {"positions": {"AAA": 2.0}, "avg_cost": {"AAA": 300.0}}}}
    live.apply_splits(state, {"stocks": splits}, "2026-10-16", {"x": ("stocks", "ml")})
    assert abs(state["accounts"]["x"]["positions"]["AAA"] - 20.0) < 1e-9
    assert abs(state["accounts"]["x"]["avg_cost"]["AAA"] - 30.0) < 1e-9
    assert state["splits"][0]["ticker"] == "AAA"


def test_a_gap_in_a_stored_day_is_filled_from_the_next_download():
    full = make_panel(60)
    first = Panel(full.open.copy(), full.close.copy(), full.volume.copy()).truncate(full.dates[49])
    first.close.iloc[-1, 0] = np.nan
    _, raw, scale, _ = data.freeze(first, None)
    panel, _, _, _ = data.freeze(full, raw, scale)
    assert panel.close.iloc[49, 0] == full.close.iloc[49, 0]


def test_a_market_with_a_ticker_missing_its_last_close_is_skipped():
    p = make_panel(50, freq="D")
    p.close.iloc[-1, 1] = np.nan
    reason = live.stale_reason("crypto", p, p.dates.max() + pd.Timedelta(days=1))
    assert reason and "BBB" in reason
