"""Live paper trading: step each fake account through any new complete days since the last run."""

from __future__ import annotations

import pandas as pd

from paper_trader import engines, simulate
from paper_trader.config import LIVE_START, MARKETS, START_CASH
from paper_trader.portfolio import Account

STALE_DAYS = {"stocks": 5, "crypto": 1}  # max calendar days between the last candle and yesterday


def init_state(accounts=None) -> dict:
    accounts = accounts or engines.ACCOUNTS
    decide = (pd.Timestamp(LIVE_START) - pd.Timedelta(days=1)).date().isoformat()
    return {
        "accounts": {
            name: {
                "cash": START_CASH,
                "positions": {},
                "avg_cost": {},
                "pending": None,
                "last_date": decide,
                "initialised": False,
            }
            for name in accounts
        },
        "skips": [],
    }


def stale_reason(market_key: str, panel, today) -> str | None:
    if panel is None or panel.close.empty:
        return "no price data from any source"
    yesterday = pd.Timestamp(today).normalize() - pd.Timedelta(days=1)
    last = panel.dates.max()
    if (yesterday - last).days > STALE_DAYS[market_key]:
        return f"latest candle is {last.date()}, expected {yesterday.date()} or close to it"
    return None


def _series(d: dict | None):
    return None if d is None else pd.Series(d, dtype=float)


def run(state: dict, panels: dict, today, markets=None, accounts=None):
    """Advance every account. Returns (state, equity rows, trade rows). Re-running on the same data is a no-op."""
    markets = markets or MARKETS
    accounts = accounts or engines.ACCOUNTS
    eq_rows, trade_rows = [], []
    for mkey, market in markets.items():
        panel = panels.get(mkey)
        reason = stale_reason(mkey, panel, today)
        if reason:
            state["skips"] = (
                state["skips"] + [{"run": str(pd.Timestamp(today).date()), "market": mkey, "reason": reason}]
            )[-30:]
            continue
        for name, (akey, kind) in accounts.items():
            if akey != mkey:
                continue
            a = state["accounts"][name]
            start = pd.Timestamp(LIVE_START)
            # Never trade before the start, and don't initialise until the first live day has closed: an account
            # set up earlier would take its first decision from a day that hadn't finished yet.
            new_days = panel.dates[(panel.dates > pd.Timestamp(a["last_date"])) & (panel.dates >= start)]
            if len(new_days) == 0:
                continue
            w, explain, _ = engines.targets(kind, market, panel, LIVE_START)
            if not a["initialised"]:
                decide = panel.dates[panel.dates < start][-1]
                row = w.loc[decide]
                a["pending"] = None if row.isna().all() else row.fillna(0.0).to_dict()
                a["initialised"] = True
                a["last_date"] = decide.date().isoformat()
            acct = Account(cash=a["cash"], positions=dict(a["positions"]))
            avg_cost = dict(a["avg_cost"])
            for d in new_days:
                rows = simulate.step(acct, d, panel, _series(a["pending"]), market.cost_rate, avg_cost)
                if rows:
                    tr = engines.attach_reasons(pd.DataFrame(rows), panel, explain)
                    trade_rows += tr.assign(account=name).to_dict("records")
                eq_rows.append({"date": d, "account": name, "equity": round(simulate.mark(acct, d, panel), 2)})
                row = w.loc[d] if d in w.index else None
                a["pending"] = None if row is None or row.isna().all() else row.fillna(0.0).to_dict()
                a["last_date"] = d.date().isoformat()
            a["cash"], a["positions"], a["avg_cost"] = acct.cash, acct.positions, avg_cost
    return state, eq_rows, trade_rows
