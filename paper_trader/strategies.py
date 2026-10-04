"""Three rule-based strategies. Each maps price history to target weights (dates x tickers).

Every row uses only data up to that row's close; tests enforce this by truncation.
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


def momentum(panel, max_positions: int) -> pd.DataFrame:
    """Hold the top names by 3-month return, skipping the most recent week."""
    close = panel.close
    mom = close.shift(5) / close.shift(68) - 1
    rank = mom.where(mom > 0).rank(axis=1, ascending=False, method="first")
    return (rank <= max_positions).astype(float) / max_positions


def mean_reversion(panel, max_positions: int) -> pd.DataFrame:
    """Buy when 5-day RSI drops below 30; sell once it recovers above 50."""
    r = rsi(panel.close, 5)
    state = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    state[r < 30] = 1.0
    state[r > 50] = 0.0
    held = state.ffill().fillna(0.0)
    denom = np.maximum(held.sum(axis=1), max_positions)
    return held.div(denom, axis=0)


def trend(panel, max_positions: int) -> pd.DataFrame:
    """Hold names above their 50-day average while the 50-day is above the 200-day."""
    close = panel.close
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    held = ((close > sma50) & (sma50 > sma200)).astype(float)
    denom = np.maximum(held.sum(axis=1), max_positions)
    return held.div(denom, axis=0)


STRATEGIES = {"momentum": momentum, "mean_reversion": mean_reversion, "trend": trend}
