"""Daily price panels. yfinance first; Coinbase public candles as the crypto fallback."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

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
            df = yf.download(
                list(tickers), start=start, auto_adjust=True, progress=False, group_by="column", threads=False
            )
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
        r = requests.get(
            url,
            params={"granularity": 86400, "start": chunk_start.isoformat(), "end": end.isoformat()},
            headers={"User-Agent": "paper-trader"},
            timeout=20,
        )
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


# A ratio between two downloads of the same day's close this far from 1 is a split (or reverse split), not a
# dividend adjustment; the live accounts' units are rescaled by it.
SPLIT_RATIO = (0.6, 1.6)


def _scaled(raw: Panel, scale: dict) -> Panel:
    k = pd.Series({t: scale.get(t, {}).get("price", 1.0) for t in raw.tickers}, dtype=float)
    v = pd.Series({t: scale.get(t, {}).get("volume", 1.0) for t in raw.tickers}, dtype=float)
    return Panel(raw.open * k, raw.close * k, raw.volume * v)


def freeze(fresh: Panel, raw: Panel | None, scale: dict | None = None):
    """The live run's price history: days already used are kept as first seen, and only new days come from `fresh`.

    Yahoo's adjusted prices move by about 1e-6 between downloads, which is enough to change the self-tuned ML
    choice night to night; frozen history makes every night's decision follow from the one before. A dividend or
    split re-bases Yahoo's whole history for a ticker, so the stored prices stay raw and a per-ticker scale
    ({"price": k, "volume": v}) is updated when the fresh download disagrees with the stored copy of the same day.

    Returns (panel, raw, scale, splits): the scaled panel to trade on, the raw rows to store (old rows unchanged, so
    the file only gains rows), the new scale, and the price ratio of each ticker that split since the last run.
    """
    scale = {t: dict(v) for t, v in (scale or {}).items()}
    if raw is None or raw.close.empty:
        return fresh, fresh, {}, {}
    splits: dict[str, float] = {}
    cur = _scaled(raw, scale)
    for t in fresh.tickers:
        if t not in cur.close.columns:
            continue
        both = cur.close[t].dropna().index.intersection(fresh.close[t].dropna().index)
        if len(both) == 0:
            continue
        day = both.max()
        ratio = float(fresh.close.at[day, t] / cur.close.at[day, t])
        if abs(ratio - 1) > 1e-3:
            sc = scale.setdefault(t, {})
            sc["price"] = sc.get("price", 1.0) * ratio
            if not SPLIT_RATIO[0] < ratio < SPLIT_RATIO[1]:
                splits[t] = ratio
                sc["volume"] = sc.get("volume", 1.0) / ratio
    cur = _scaled(raw, scale)
    k = pd.Series({t: scale.get(t, {}).get("price", 1.0) for t in fresh.tickers}, dtype=float)
    v = pd.Series({t: scale.get(t, {}).get("volume", 1.0) for t in fresh.tickers}, dtype=float)
    last = raw.close.index.max()
    new = fresh.close.index[fresh.close.index > last]
    raw_out = []
    for raw_f, fresh_f, div in ((raw.open, fresh.open, k), (raw.close, fresh.close, k), (raw.volume, fresh.volume, v)):
        raw_f = raw_f.reindex(columns=fresh.tickers)
        # A gap in a stored day (one ticker missing that night) is filled from the fresh download.
        fill = (fresh_f.reindex(raw_f.index) / div).where(raw_f.isna())
        raw_out.append(pd.concat([raw_f.fillna(fill), fresh_f.loc[new] / div]).astype(float))
    raw_new = Panel(*raw_out)
    return _scaled(raw_new, scale), raw_new, scale, splits


def load_frozen(path: Path) -> tuple[Panel | None, dict]:
    """The stored raw prices and the per-ticker scale beside them."""
    if not path.exists():
        return None, {}
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    panel = Panel(*(df[f].astype(float) for f in ("open", "close", "volume")))
    scale_path = path.with_name(path.name.replace(".csv", ".scale.json"))
    return panel, (json.loads(scale_path.read_text()) if scale_path.exists() else {})


def save_frozen(path: Path, raw: Panel, scale: dict) -> None:
    pd.concat({"open": raw.open, "close": raw.close, "volume": raw.volume}, axis=1).to_csv(path)
    scale_path = path.with_name(path.name.replace(".csv", ".scale.json"))
    scale_path.write_text(json.dumps(scale, indent=2, sort_keys=True) + "\n")
