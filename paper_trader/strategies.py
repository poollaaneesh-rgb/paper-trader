"""Three rule-based strategies. Each maps price history to target weights (dates x tickers).

Every row uses only data up to that row's close; tests enforce this by truncation. The parameters default to the
original settings; the self-tuning layer (variants.py) tries slower and faster versions of each.
"""

import numpy as np
import pandas as pd


def rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def momentum(panel, max_positions: int, lookback: int = 63, skip: int = 5) -> pd.DataFrame:
    """Hold the top names by `lookback`-day return, skipping the most recent `skip` days (short-term reversal)."""
    close = panel.close
    mom = close.shift(skip) / close.shift(skip + lookback) - 1
    rank = mom.where(mom > 0).rank(axis=1, ascending=False, method="first")
    return (rank <= max_positions).astype(float) / max_positions


def mean_reversion(
    panel, max_positions: int, rsi_n: int = 5, buy_below: float = 30, sell_above: float = 50
) -> pd.DataFrame:
    """Buy when the RSI drops below `buy_below`; sell once it recovers above `sell_above`."""
    r = rsi(panel.close, rsi_n)
    state = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    state[r < buy_below] = 1.0
    state[r > sell_above] = 0.0
    held = state.ffill().fillna(0.0)
    denom = np.maximum(held.sum(axis=1), max_positions)
    return held.div(denom, axis=0)


def trend(panel, max_positions: int, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Hold names above their `fast`-day average while that average is above the `slow`-day one."""
    close = panel.close
    sma_fast = close.rolling(fast, min_periods=fast).mean()
    sma_slow = close.rolling(slow, min_periods=slow).mean()
    held = ((close > sma_fast) & (sma_fast > sma_slow)).astype(float)
    denom = np.maximum(held.sum(axis=1), max_positions)
    return held.div(denom, axis=0)


STRATEGIES = {"momentum": momentum, "mean_reversion": mean_reversion, "trend": trend}
