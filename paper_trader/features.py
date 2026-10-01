"""Per-asset daily features, all computed from data up to that day's close."""
import pandas as pd

from paper_trader.strategies import rsi

FEATURES = ["ret1", "ret5", "ret20", "ret60", "vol20", "rsi14", "dist50", "dist200", "volchg"]


def build(panel) -> pd.DataFrame:
    close, volume = panel.close, panel.volume
    ret1 = close.pct_change(fill_method=None)
    frames = {
        "ret1": ret1,
        "ret5": close.pct_change(5, fill_method=None),
        "ret20": close.pct_change(20, fill_method=None),
        "ret60": close.pct_change(60, fill_method=None),
        "vol20": ret1.rolling(20, min_periods=20).std(),
        "rsi14": rsi(close, 14) / 100,
        "dist50": close / close.rolling(50, min_periods=50).mean() - 1,
        "dist200": close / close.rolling(200, min_periods=200).mean() - 1,
        "volchg": volume / volume.rolling(20, min_periods=20).mean() - 1,
    }
    long = pd.concat({k: v.stack() for k, v in frames.items()}, axis=1)
    long.index.names = ["date", "ticker"]
    return long[FEATURES]


def labels(panel) -> pd.Series:
    """1 if the open-to-open return from t+1 to t+2 is positive (the holding period of a decision at t)."""
    fwd = panel.open.shift(-2) / panel.open.shift(-1) - 1
    y = (fwd > 0).astype(float).where(fwd.notna())
    s = y.stack()
    s.index.names = ["date", "ticker"]
    return s
