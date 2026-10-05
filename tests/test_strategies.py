import numpy as np
import pandas as pd
import pytest

from paper_trader import strategies, tournament

FUNCS = [strategies.momentum, strategies.mean_reversion, strategies.trend]


@pytest.mark.parametrize("fn", FUNCS)
def test_no_lookahead(panel, fn):
    full = fn(panel, 2)
    for t in panel.dates[[250, 300, 399]]:
        part = fn(panel.truncate(t), 2)
        pd.testing.assert_series_equal(full.loc[t], part.loc[t], check_names=False)


@pytest.mark.parametrize("fn", FUNCS)
def test_weights_are_valid(panel, fn):
    w = fn(panel, 2)
    assert (w.fillna(0) >= 0).all().all()
    assert (w.fillna(0).sum(axis=1) <= 1 + 1e-9).all()


def test_tournament_no_lookahead(panel):
    full, _ = tournament.weights(panel, 2)
    for t in panel.dates[[250, 330, 399]]:
        part, _ = tournament.weights(panel.truncate(t), 2)
        pd.testing.assert_series_equal(full.loc[t], part.loc[t], check_names=False)


def test_allocations_floor_and_monthly(panel):
    w, alloc = tournament.weights(panel, 2)
    assert np.allclose(alloc.sum(axis=1), 1.0)
    assert (alloc >= 0.10 - 1e-12).all().all()
    months = pd.Series(alloc.index.to_period("M"), index=alloc.index)
    month_start = months != months.shift(1)
    changed = alloc.diff().abs().sum(axis=1) > 1e-12
    assert not (changed & ~month_start).any()
    assert (w.sum(axis=1) <= 1 + 1e-9).all()


def _panel_from_close(close: pd.DataFrame):
    from paper_trader.data import Panel

    return Panel(open=close.copy(), close=close, volume=close * 0 + 1000)


def test_momentum_lookback_sets_the_ranking_window():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    a = np.r_[np.full(70, 100.0), np.linspace(100, 130, 10)]  # flat, then a sharp rise in the last 10 days
    b = np.linspace(100, 160, 80)  # a bigger rise, spread evenly over the whole period
    close = pd.DataFrame({"AAA": a, "BBB": b}, index=dates)
    short = strategies.momentum(_panel_from_close(close), 1, lookback=5, skip=0)
    long = strategies.momentum(_panel_from_close(close), 1, lookback=60, skip=0)
    assert short.iloc[-1]["AAA"] == 1.0 and short.iloc[-1]["BBB"] == 0.0
    assert long.iloc[-1]["BBB"] == 1.0 and long.iloc[-1]["AAA"] == 0.0


def test_mean_reversion_thresholds_decide_what_is_held(panel):
    everything = strategies.mean_reversion(panel, 2, rsi_n=5, buy_below=101, sell_above=102)
    nothing = strategies.mean_reversion(panel, 2, rsi_n=5, buy_below=-1, sell_above=50)
    assert (everything.iloc[-1] > 0).all()
    assert (nothing.fillna(0) == 0).all().all()


def test_trend_averages_follow_the_parameters():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    close = pd.DataFrame({"UP": np.linspace(100, 130, 30), "DOWN": np.linspace(130, 100, 30)}, index=dates)
    w = strategies.trend(_panel_from_close(close), 1, fast=2, slow=3)
    assert w.iloc[-1]["UP"] == 1.0 and w.iloc[-1]["DOWN"] == 0.0
    assert w.iloc[1].isna().all() or (w.iloc[1] == 0).all()  # before the slow average exists, nothing is held


def test_allocations_cadence_and_lookback():
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(120, 3)), index=dates, columns=["a", "b", "c"])
    weekly = tournament.allocations(rets, lookback=20, cadence="W")
    weeks = pd.Series(weekly.index.to_period("W"), index=weekly.index)
    changed = weekly.diff().abs().sum(axis=1) > 1e-12
    assert not (changed & ~(weeks != weeks.shift(1))).any()
    assert changed.any()
    daily = tournament.allocations(rets, lookback=20, cadence="D")
    assert (daily.diff().abs().sum(axis=1) > 1e-12).sum() > changed.sum()
    late = tournament.allocations(rets, lookback=100, cadence="D")
    assert np.allclose(late.iloc[:99].to_numpy(), 1 / 3)


def test_tournament_weights_accept_strategy_parameters(panel):
    default, _ = tournament.weights(panel, 2)
    custom, alloc = tournament.weights(
        panel, 2, params={"momentum": {"lookback": 21}}, lookback=21, cadence="W", cost_rate=0.005
    )
    assert custom.shape == default.shape
    assert not custom.fillna(0).equals(default.fillna(0))
    assert np.allclose(alloc.sum(axis=1), 1.0)
