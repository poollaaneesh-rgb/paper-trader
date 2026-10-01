"""The six fake accounts: which market, which engine, and why each trade happened."""
from __future__ import annotations

import numpy as np
import pandas as pd

from paper_trader import ml, tournament
from paper_trader.config import MARKETS, ML_THRESHOLD
from paper_trader.strategies import STRATEGIES, rsi

ACCOUNTS = {
    "tournament_stocks": ("stocks", "tournament"),
    "tournament_crypto": ("crypto", "tournament"),
    "ml_stocks": ("stocks", "ml"),
    "ml_crypto": ("crypto", "ml"),
    "bench_spy": ("stocks", "bench"),
    "bench_btc": ("crypto", "bench"),
}


def targets(kind: str, market, panel, start):
    """Return (weights, explain(decision_date, ticker, side) -> str, diagnostics)."""
    start = pd.Timestamp(start)
    if kind == "bench":
        w = pd.DataFrame(np.nan, index=panel.dates, columns=panel.tickers)
        decide = panel.dates[panel.dates < start][-1]
        w.loc[decide] = 0.0
        w.loc[decide, market.benchmark] = 1.0
        return w, lambda d, t, side: "buy and hold, never trades again", {}

    if kind == "tournament":
        combined, alloc = tournament.weights(panel, market.max_positions)
        per = {name: fn(panel, market.max_positions) for name, fn in STRATEGIES.items()}
        close = panel.close
        mom = close.shift(5) / close.shift(68) - 1
        rsi5 = rsi(close, 5)
        dist50 = close / close.rolling(50, min_periods=50).mean() - 1
        signal = {"momentum": lambda d, t: f"3-mo return {mom.at[d, t]:+.1%}",
                  "mean_reversion": lambda d, t: f"5-day RSI {rsi5.at[d, t]:.0f}",
                  "trend": lambda d, t: f"{dist50.at[d, t]:+.1%} vs 50-day avg"}

        def explain(d, t, side):
            parts = [f"{name.replace('_', ' ')} wants {per[name].at[d, t] * alloc.at[d, name]:.0%} "
                     f"({signal[name](d, t)})" for name in per if per[name].at[d, t] > 0]
            if parts:
                return "; ".join(parts)
            return "no strategy holds it any more"

        last_alloc = {k: round(float(v), 3) for k, v in alloc.iloc[-1].items()}
        return combined, explain, {"allocations": last_alloc, "allocation_history": alloc}

    w, diag = ml.weights(panel, market.max_positions, start=panel.dates[panel.dates < start][-1])
    probs = diag.pop("probs")

    def explain(d, t, side):
        p = probs.at[d, t] if d in probs.index else np.nan
        if np.isnan(p):
            return "no model yet (needs a year of training data)"
        if side == "buy":
            return f"model gives {p:.0%} odds of a rise (threshold {ML_THRESHOLD:.0%})"
        return f"model odds fell to {p:.0%} or it dropped out of the top {market.max_positions}"

    return w, explain, diag


def attach_reasons(trades: pd.DataFrame, panel, explain) -> pd.DataFrame:
    """A trade filled on day d was decided at the close of the previous trading day."""
    if trades.empty:
        return trades.assign(reason=pd.Series(dtype=str))
    dates = panel.dates
    prev = {d: dates[i - 1] for i, d in enumerate(dates) if i > 0}
    trades = trades.copy()
    trades["reason"] = [explain(prev[r.date], r.ticker, r.side) for r in trades.itertuples()]
    return trades
