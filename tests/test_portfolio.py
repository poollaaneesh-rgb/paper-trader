import math

import pandas as pd

from paper_trader.portfolio import Account


def prices(**kw):
    return pd.Series(kw, dtype=float)


def test_round_trip_loses_exact_costs():
    acct = Account(cash=400.0)
    p = prices(AAA=50.0)
    acct.rebalance(pd.Series({"AAA": 1.0}), p, cost_rate=0.005)
    invested = acct.positions["AAA"] * 50.0
    acct.rebalance(pd.Series(dtype=float), p, cost_rate=0.005)
    assert "AAA" not in acct.positions
    assert math.isclose(acct.cash, 400.0 - 2 * 0.005 * invested, abs_tol=1e-9)


def test_cash_never_negative():
    acct = Account(cash=400.0)
    p = prices(AAA=10.0, BBB=20.0)
    acct.rebalance(pd.Series({"AAA": 0.5, "BBB": 0.5}), p, cost_rate=0.01)
    assert acct.cash >= 0
    acct.rebalance(pd.Series({"AAA": 1.0}), prices(AAA=12.0, BBB=15.0), cost_rate=0.01)
    assert acct.cash >= 0


def test_small_trades_skipped():
    acct = Account(cash=400.0)
    p = prices(AAA=10.0)
    acct.rebalance(pd.Series({"AAA": 0.5}), p, cost_rate=0.0)
    trades = acct.rebalance(pd.Series({"AAA": 0.505}), p, cost_rate=0.0)
    assert trades == []


def test_nan_price_leaves_position_untouched():
    acct = Account(cash=400.0)
    acct.rebalance(pd.Series({"AAA": 0.5, "BBB": 0.5}), prices(AAA=10.0, BBB=10.0), cost_rate=0.0)
    before = acct.positions["BBB"]
    acct.rebalance(pd.Series({"AAA": 1.0}), prices(AAA=10.0, BBB=float("nan")), cost_rate=0.0)
    assert acct.positions["BBB"] == before


def test_equity_marks_positions():
    acct = Account(cash=100.0, positions={"AAA": 2.0})
    assert acct.equity(prices(AAA=50.0)) == 200.0
