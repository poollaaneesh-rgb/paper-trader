# Paper Trader, Bloomberg check: the stock tournament on Bloomberg prices vs Yahoo prices.
#
# Paste this whole file into a BQuant notebook (BQNT <GO>) in the FAR Lab and run it.
# Locally, `python bquant/tournament_close.py` runs the identical code on the frozen Yahoo prices.
# Only summary numbers leave the Terminal (write them in bquant/README.md); never export raw data.
#
# Simplified on purpose so both sources run the same code: closes only, fills at the next close,
# costs charged on weight turnover. Compare the two columns with each other, not with the main
# backtest (which fills at the open).

import numpy as np
import pandas as pd

TICKERS = {  # Yahoo -> Bloomberg
    "SPY": "SPY US Equity", "QQQ": "QQQ US Equity", "IWM": "IWM US Equity", "TLT": "TLT US Equity",
    "GLD": "GLD US Equity", "XLK": "XLK US Equity", "XLF": "XLF US Equity", "XLE": "XLE US Equity",
    "XLV": "XLV US Equity", "XLY": "XLY US Equity", "AAPL": "AAPL US Equity", "MSFT": "MSFT US Equity",
    "NVDA": "NVDA US Equity", "AMZN": "AMZN US Equity", "GOOGL": "GOOGL US Equity", "META": "META US Equity",
    "BRK-B": "BRK/B US Equity", "JPM": "JPM US Equity", "V": "V US Equity", "UNH": "UNH US Equity",
    "XOM": "XOM US Equity", "JNJ": "JNJ US Equity", "PG": "PG US Equity", "HD": "HD US Equity",
    "COST": "COST US Equity", "AVGO": "AVGO US Equity", "LLY": "LLY US Equity", "WMT": "WMT US Equity",
    "MA": "MA US Equity", "KO": "KO US Equity",
}
DATA_START, START, END = "2015-01-01", "2018-01-01", "2026-09-30"
CASH, COST, MAX_POS, FLOOR, LOOKBACK = 400.0, 0.0005, 5, 0.10, 63


# ---------- data ----------
def bloomberg_closes() -> pd.DataFrame:
    """Daily PX_LAST from BQL, one column per Yahoo ticker. Runs only inside BQuant."""
    import bql
    bq = bql.Service()
    item = bq.data.px_last(dates=bq.func.range(DATA_START, END), fill="prev")
    res = bq.execute(bql.Request(list(TICKERS.values()), {"close": item}))
    df = res[0].df().reset_index()
    wide = df.pivot(index="DATE", columns="ID", values="close")
    wide.index = pd.to_datetime(wide.index)
    return wide.rename(columns={v: k for k, v in TICKERS.items()})[list(TICKERS)].astype(float)


def yahoo_closes() -> pd.DataFrame:
    """The frozen Yahoo closes from the main backtest (local only)."""
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "results" / "backtest" / "prices_stocks.csv.gz"
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    return df["close"].loc[DATA_START:END].astype(float)


def check(close: pd.DataFrame) -> None:
    """Flag one-day moves over 35%: usually an unadjusted split, which would poison every signal."""
    rets = close.pct_change(fill_method=None)
    big = rets.abs().stack()
    big = big[big > 0.35]
    print(f"{close.shape[1]} tickers, {close.index.min().date()} to {close.index.max().date()}, "
          f"{int(close.isna().sum().sum())} missing values")
    print("No suspicious jumps." if big.empty else f"Check these jumps (splits?):\n{big.head(20)}")


# ---------- strategies (same rules as paper_trader/strategies.py, closes only) ----------
def rsi(close, n):
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return out.where(loss != 0, 100.0).where(gain.notna())


def momentum(close):
    mom = close.shift(5) / close.shift(68) - 1
    rank = mom.where(mom > 0).rank(axis=1, ascending=False, method="first")
    return (rank <= MAX_POS).astype(float) / MAX_POS


def mean_reversion(close):
    r = rsi(close, 5)
    state = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    state[r < 30], state[r > 50] = 1.0, 0.0
    held = state.ffill().fillna(0.0)
    return held.div(np.maximum(held.sum(axis=1), MAX_POS), axis=0)


def trend(close):
    s50, s200 = close.rolling(50, min_periods=50).mean(), close.rolling(200, min_periods=200).mean()
    held = ((close > s50) & (s50 > s200)).astype(float)
    return held.div(np.maximum(held.sum(axis=1), MAX_POS), axis=0)


def tournament(close):
    """Monthly reweighting toward the best trailing 3-month Sharpe, with a 10% floor each."""
    rets = close.pct_change(fill_method=None)
    per = {"momentum": momentum(close), "mean_reversion": mean_reversion(close), "trend": trend(close)}
    sret = pd.DataFrame({k: (w.shift(2) * rets).sum(axis=1, min_count=1).fillna(0) for k, w in per.items()})
    months = pd.Series(sret.index.to_period("M"), index=sret.index)
    alloc = pd.DataFrame(np.nan, index=sret.index, columns=sret.columns)
    alloc.iloc[0] = 1 / 3
    for t in sret.index[(months != months.shift(1)).to_numpy()]:
        win = sret.loc[:t].tail(LOOKBACK)
        sd = win.std().replace(0, np.nan)
        score = (win.mean() / sd * np.sqrt(252)).fillna(0).clip(lower=0) if len(win) == LOOKBACK else 0 * win.mean()
        alloc.loc[t] = 1 / 3 if score.sum() == 0 else FLOOR + (1 - 3 * FLOOR) * score / score.sum()
    alloc = alloc.ffill()
    return sum(per[k].mul(alloc[k], axis=0) for k in per)


# ---------- simulation and scorecard ----------
def run(close, weights):
    """Decide at close t, hold from close t+1: earns returns from t+2. Costs on weight turnover."""
    rets = close.pct_change(fill_method=None).fillna(0)
    held = weights.shift(2).fillna(0)
    turnover = (weights.shift(1) - weights.shift(2)).abs().sum(axis=1).fillna(0)
    daily = (held * rets).sum(axis=1) - COST * turnover
    daily = daily.loc[START:END]
    return CASH * (1 + daily).cumprod()


def score(eq):
    r = eq.pct_change().dropna()
    return {"final": round(float(eq.iloc[-1]), 2), "return": f"{eq.iloc[-1] / CASH - 1:+.1%}",
            "worst_drop": f"{(eq / eq.cummax() - 1).min():.1%}",
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2)}


def report(close, source):
    check(close)
    hold = pd.DataFrame({"SPY": 1.0}, index=close.index).reindex(columns=close.columns, fill_value=0.0)
    rows = {"Tournament": score(run(close, tournament(close))), "SPY hold": score(run(close, hold))}
    print(f"\n{source} prices, {START} to {END}, ${CASH:.0f} start:")
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    try:
        import bql  # noqa: F401  (present only inside BQuant)
        report(bloomberg_closes(), "Bloomberg")
    except ImportError:
        report(yahoo_closes(), "Yahoo (frozen)")
