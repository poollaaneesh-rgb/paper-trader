"""Scorecard numbers for one account."""

import numpy as np
import pandas as pd


def summary(equity: pd.Series, trades: pd.DataFrame, periods: int, start_cash: float) -> dict:
    if equity.empty:
        return {
            "final_equity": start_cash,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": None,
            "win_rate": None,
            "n_trades": 0,
            "days": 0,
        }
    rets = equity.pct_change().dropna()
    sd = rets.std()
    pnl = trades["realized_pnl"].dropna() if "realized_pnl" in trades else pd.Series(dtype=float)
    return {
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return": float(equity.iloc[-1] / start_cash - 1),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "sharpe": float(rets.mean() / sd * np.sqrt(periods)) if sd and sd > 0 else None,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else None,
        "n_trades": len(trades),
        "days": len(equity),
    }


def trailing_return(equity: pd.Series, days: int = 30) -> float | None:
    """Return over the last `days` calendar days, from the last mark on or before their start; None until then.

    The scoreboard that picks which bot holds the real money ranks on this (the user's rule, 2026-10-08).
    """
    if equity.empty:
        return None
    equity = equity.sort_index()
    start = equity.index[-1] - pd.Timedelta(days=days)
    before = equity[equity.index <= start]
    if before.empty:
        return None
    return float(equity.iloc[-1] / before.iloc[-1] - 1)
