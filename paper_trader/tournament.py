"""Strategy tournament: capital shifts toward the strategies that have been winning.

The original rules (and the `*_fixed` accounts): three strategies at their default settings, re-weighted on the
first trading day of each month by trailing 63-day Sharpe ratio of gross returns, 10% floor each. The self-tuning
layer tries other lookbacks, cadences, strategy settings and scoring net of costs (variants.py).
"""

import numpy as np
import pandas as pd

from paper_trader.config import TOURNAMENT_FLOOR
from paper_trader.strategies import STRATEGIES

LOOKBACK = 63  # about three months of trading days


def open_returns(panel) -> pd.DataFrame:
    return panel.open.pct_change(fill_method=None)


def strategy_returns(weights: pd.DataFrame, panel, cost_rate: float = 0.0) -> pd.Series:
    """Weights decided at close t are held from open t+1 to open t+2, so they earn R[t+2].

    With a cost rate, the turnover between consecutive target rows is charged on the day those weights start earning.
    """
    gross = (weights.shift(2) * open_returns(panel)).sum(axis=1, min_count=1).fillna(0.0)
    if not cost_rate:
        return gross
    turnover = (weights.fillna(0.0) - weights.fillna(0.0).shift(1)).abs().sum(axis=1).shift(2).fillna(0.0)
    return gross - cost_rate * turnover


def _period_starts(index: pd.DatetimeIndex, cadence: str) -> pd.DatetimeIndex:
    if cadence == "D":
        return index
    periods = pd.Series(index.to_period(cadence), index=index)
    return index[(periods != periods.shift(1)).to_numpy()]


def allocations(rets: pd.DataFrame, lookback: int = LOOKBACK, cadence: str = "M") -> pd.DataFrame:
    """Share of capital per strategy, set at each period start from the trailing Sharpe ratio, with a floor each."""
    n = rets.shape[1]
    starts = _period_starts(rets.index, cadence)
    alloc = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns)
    alloc.iloc[0] = 1.0 / n
    for t in starts:
        window = rets.loc[:t].tail(lookback)
        if len(window) < lookback:
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


def weights(
    panel,
    max_positions: int,
    params: dict[str, dict] | None = None,
    lookback: int = LOOKBACK,
    cadence: str = "M",
    cost_rate: float = 0.0,
):
    """Return (combined target weights, allocations per strategy).

    `params` holds keyword arguments per strategy (e.g. {"momentum": {"lookback": 126}}); `cost_rate` > 0 scores the
    strategies net of their own trading costs.
    """
    params = params or {}
    per = {name: fn(panel, max_positions, **params.get(name, {})) for name, fn in STRATEGIES.items()}
    rets = pd.DataFrame({name: strategy_returns(w, panel, cost_rate) for name, w in per.items()})
    alloc = allocations(rets, lookback, cadence)
    combined = sum(per[name].mul(alloc[name], axis=0) for name in per)
    return combined, alloc
