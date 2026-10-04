"""Strategy tournament: capital shifts monthly toward the strategies that have been winning."""

import numpy as np
import pandas as pd

from paper_trader.config import TOURNAMENT_FLOOR
from paper_trader.strategies import STRATEGIES

LOOKBACK = 63  # about three months of trading days


def open_returns(panel) -> pd.DataFrame:
    return panel.open.pct_change(fill_method=None)


def strategy_returns(weights: pd.DataFrame, panel) -> pd.Series:
    """Weights decided at close t are held from open t+1 to open t+2, so they earn R[t+2]."""
    return (weights.shift(2) * open_returns(panel)).sum(axis=1, min_count=1).fillna(0.0)


def allocations(rets: pd.DataFrame) -> pd.DataFrame:
    n = rets.shape[1]
    months = pd.Series(rets.index.to_period("M"), index=rets.index)
    starts = rets.index[(months != months.shift(1)).to_numpy()]
    alloc = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns)
    alloc.iloc[0] = 1.0 / n
    for t in starts:
        window = rets.loc[:t].tail(LOOKBACK)
        if len(window) < LOOKBACK:
            alloc.loc[t] = 1.0 / n
            continue
        std = window.std()
        sharpe = (window.mean() / std.replace(0, np.nan) * np.sqrt(252)).fillna(0.0)
        score = sharpe.clip(lower=0)
        if score.sum() == 0:
            alloc.loc[t] = 1.0 / n
        else:
            alloc.loc[t] = TOURNAMENT_FLOOR + (1 - n * TOURNAMENT_FLOOR) * score / score.sum()
    return alloc.ffill()


def weights(panel, max_positions: int):
    """Return (combined target weights, monthly allocations per strategy)."""
    per = {name: fn(panel, max_positions) for name, fn in STRATEGIES.items()}
    rets = pd.DataFrame({name: strategy_returns(w, panel) for name, w in per.items()})
    alloc = allocations(rets)
    combined = sum(per[name].mul(alloc[name], axis=0) for name in per)
    return combined, alloc
