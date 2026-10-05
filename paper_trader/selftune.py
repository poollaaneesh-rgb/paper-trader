"""The self-tuning layer: shadow variants of an engine, scored after costs, followed when one leads clearly.

Every variant is a table of target weights computed from price history alone, so the layer is itself a pure
function of prices: run it on data truncated at day t and the choice for t does not change (tests enforce this).
Each day it scores every variant on its trailing WINDOW-day return after costs and moves to the leader only when
the leader is ahead of the current choice by MARGIN. The three numbers were fixed before any results were seen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 126  # trading days of shadow results a variant is judged on (about six months)
MARGIN = 0.02  # the leader must be ahead by this much, in log return over the window, to replace the current choice
WARMUP = 2 * WINDOW  # shadow history computed before a live start, so the first live choice has a real record behind it


def net_returns(weights: pd.DataFrame, panel, cost_rate: float) -> pd.Series:
    """Daily return of a variant: weights decided at close t earn open t+1 to open t+2, less costs on the turnover."""
    w = weights.reindex(index=panel.dates, columns=panel.tickers).fillna(0.0)
    opens = panel.open.pct_change(fill_method=None)
    gross = (w.shift(2) * opens).sum(axis=1, min_count=1).fillna(0.0)
    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1).shift(2).fillna(0.0)
    return gross - cost_rate * turnover


def scores(returns: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Trailing log return per variant; NaN until a full window exists."""
    return np.log1p(returns.clip(lower=-0.99)).rolling(window, min_periods=window).sum()


def choose(returns: pd.DataFrame, default: str, window: int = WINDOW, margin: float = MARGIN) -> pd.Series:
    """The variant in force at each close: the default until a leader is ahead by the margin, then that leader."""
    s = scores(returns, window).to_numpy()
    cols = list(returns.columns)
    current = cols.index(default)
    out = []
    for row in s:
        if not np.isnan(row).all():
            lead = int(np.nanargmax(row))
            if lead != current and row[lead] - (row[current] if not np.isnan(row[current]) else -np.inf) > margin:
                current = lead
        out.append(cols[current])
    return pd.Series(out, index=returns.index, name="variant")


def combine(variants: dict[str, pd.DataFrame], chosen: pd.Series) -> pd.DataFrame:
    """Target weights that follow the chosen variant day by day."""
    first = next(iter(variants.values()))
    out = pd.DataFrame(0.0, index=chosen.index, columns=first.columns)
    for name in chosen.unique():
        days = chosen.index[chosen == name]
        out.loc[days] = variants[name].reindex(index=days, columns=out.columns).fillna(0.0)
    return out


def run(variants: dict[str, pd.DataFrame], panel, cost_rate: float, default: str, window=WINDOW, margin=MARGIN):
    """Return (weights, chosen variant per day, scores per variant per day)."""
    rets = pd.DataFrame({name: net_returns(w, panel, cost_rate) for name, w in variants.items()})
    first = min(w.index[0] for w in variants.values())
    rets = rets.loc[first:]
    chosen = choose(rets, default, window, margin)
    return combine(variants, chosen), chosen, scores(rets, window)
