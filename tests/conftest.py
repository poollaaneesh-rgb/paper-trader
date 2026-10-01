import numpy as np
import pandas as pd
import pytest

from paper_trader.data import Panel


def make_panel(n_days=400, tickers=("AAA", "BBB", "CCC", "DDD"), seed=0, freq="B"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq=freq)
    rets = rng.normal(0.0005, 0.02, size=(n_days, len(tickers)))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=list(tickers))
    opn = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.003, size=close.shape))
    vol = pd.DataFrame(rng.integers(1_000, 10_000, size=close.shape).astype(float), index=dates, columns=list(tickers))
    return Panel(open=opn, close=close, volume=vol)


@pytest.fixture
def panel():
    return make_panel()
