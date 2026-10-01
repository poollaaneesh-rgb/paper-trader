"""Daily price panels. yfinance first; Coinbase public candles as the crypto fallback."""
from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import requests


@dataclass
class Panel:
    open: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def truncate(self, end) -> Panel:
        end = pd.Timestamp(end)
        return Panel(self.open.loc[:end], self.close.loc[:end], self.volume.loc[:end])


def complete_days(panel: Panel, today_utc) -> Panel:
    """Keep only days strictly before today's UTC date: today's candle is still forming."""
    last = pd.Timestamp(today_utc).normalize() - pd.Timedelta(days=1)
    return panel.truncate(last)


def _yfinance(tickers, start) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    for attempt in range(3):
        try:
            df = yf.download(list(tickers), start=start, auto_adjust=True, progress=False,
                             group_by="column", threads=False)
            if not df.empty:
                out = {}
                for field in ("Open", "Close", "Volume"):
                    frame = df[field].copy()
                    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
                    out[field.lower()] = frame.reindex(columns=list(tickers))
                return out
        except Exception as exc:  # network or parsing failure: retry, then give up
            print(f"yfinance attempt {attempt + 1} failed: {exc}")
        time.sleep(5 * (attempt + 1))
    return {}


def coinbase_candles(product: str, start) -> pd.DataFrame:
    """Daily candles from Coinbase's public API (no key). Columns: open, close, volume."""
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    end = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    start = pd.Timestamp(start)
    rows = []
    while end > start:
        chunk_start = max(start, end - pd.Timedelta(days=299))
        r = requests.get(url, params={"granularity": 86400, "start": chunk_start.isoformat(),
                                      "end": end.isoformat()},
                         headers={"User-Agent": "paper-trader"}, timeout=20)
        if r.status_code != 200:
            break
        rows.extend(r.json())
        end = chunk_start
        time.sleep(0.4)
    if not rows:
        return pd.DataFrame(columns=["open", "close", "volume"])
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df.index = pd.to_datetime(df["time"], unit="s").dt.normalize()
    df = df[~df.index.duplicated()].sort_index()
    return df[["open", "close", "volume"]].astype(float)


def fetch(market, start) -> Panel:
    frames = _yfinance(market.tickers, start)
    if not frames:
        frames = {k: pd.DataFrame(columns=list(market.tickers), dtype=float) for k in ("open", "close", "volume")}
    if market.name == "crypto":
        yesterday = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None) - pd.Timedelta(days=1)
        for t in market.tickers:
            col = frames["close"][t].dropna() if t in frames["close"] else pd.Series(dtype=float)
            if col.empty or col.index.max() < yesterday:
                cb = coinbase_candles(t, start)
                if cb.empty:
                    continue
                print(f"{t}: filled from Coinbase")
                for k in ("open", "close", "volume"):
                    merged = frames[k].reindex(frames[k].index.union(cb.index))
                    merged[t] = merged[t].combine_first(cb[k])
                    frames[k] = merged
    idx = frames["close"].dropna(how="all").index
    return Panel(*(frames[k].reindex(idx).astype(float) for k in ("open", "close", "volume")))
