"""Step an account through the days: decisions made at close t fill at the open of t+1."""
from __future__ import annotations

import pandas as pd

from paper_trader.portfolio import Account


def step(account: Account, day, panel, target: pd.Series | None, cost_rate: float,
         avg_cost: dict | None = None) -> list[dict]:
    """Fill a pending target at `day`'s open. `None` (or an all-NaN row) means hold."""
    if target is None or target.isna().all():
        return []
    day = pd.Timestamp(day)
    opens = panel.open.loc[day]
    last_close = panel.close.loc[:day].ffill().iloc[-2] if len(panel.close.loc[:day]) > 1 else None
    avg_cost = {} if avg_cost is None else avg_cost
    rows = []
    for t in account.rebalance(target, opens, cost_rate, value_prices=last_close):
        pnl = None
        if t.units < 0:
            basis = avg_cost.get(t.ticker, t.price)
            pnl = (t.price - basis) * -t.units - t.cost
            if t.ticker not in account.positions:
                avg_cost.pop(t.ticker, None)
        else:
            held_before = account.positions[t.ticker] - t.units
            basis = avg_cost.get(t.ticker, 0.0)
            avg_cost[t.ticker] = (basis * held_before + t.price * t.units + t.cost) / account.positions[t.ticker]
        rows.append({"date": day, "ticker": t.ticker, "side": "buy" if t.units > 0 else "sell",
                     "units": abs(t.units), "price": t.price, "value": abs(t.units) * t.price,
                     "cost": t.cost, "realized_pnl": pnl})
    return rows


def mark(account: Account, day, panel) -> float:
    closes = panel.close.loc[:pd.Timestamp(day)].ffill().iloc[-1]
    return account.equity(closes)


def run(panel, weights: pd.DataFrame, start, end, cash: float, cost_rate: float):
    """Simulate from `start` to `end`. The decision at the last close before `start` fills on day one."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = panel.dates[(panel.dates >= start) & (panel.dates <= end)]
    before = weights.index[weights.index < start]
    pending = weights.loc[before[-1]] if len(before) else None
    account, avg_cost, equity, trades = Account(cash=cash), {}, {}, []
    for d in days:
        trades += step(account, d, panel, pending, cost_rate, avg_cost)
        equity[d] = mark(account, d, panel)
        pending = weights.loc[d] if d in weights.index else None
    cols = ["date", "ticker", "side", "units", "price", "value", "cost", "realized_pnl"]
    return pd.Series(equity, dtype=float), pd.DataFrame(trades, columns=cols), account
