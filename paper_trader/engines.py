"""The six fake accounts: which market, which engine, and why each trade happened.

Since 2026-10-05 the tournament and ML engines are self-tuning: each runs a menu of settings as shadow accounts
(variants.py) and follows the one that leads after costs (selftune.py). The original fixed rules stay in the menu
as its default entry, and the backtest reports them beside the tuned result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from paper_trader import ml, selftune, variants
from paper_trader.config import ML_THRESHOLD
from paper_trader.strategies import STRATEGIES

ACCOUNTS = {
    "tournament_stocks": ("stocks", "tournament"),
    "tournament_crypto": ("crypto", "tournament"),
    "ml_stocks": ("stocks", "ml"),
    "ml_crypto": ("crypto", "ml"),
    "bench_spy": ("stocks", "bench"),
    "bench_btc": ("crypto", "bench"),
    "bench_eq_stocks": ("stocks", "bench_eq"),
    "bench_eq_crypto": ("crypto", "bench_eq"),
}


def _switches(chosen: pd.Series, start) -> int:
    live = chosen.loc[pd.Timestamp(start) :]
    return int((live != live.shift(1)).sum() - 1) if len(live) else 0


def targets(kind: str, market, panel, start):
    """Return (weights, explain(decision_date, ticker, side) -> str, diagnostics)."""
    start = pd.Timestamp(start)
    if kind == "bench":
        w = pd.DataFrame(np.nan, index=panel.dates, columns=panel.tickers)
        decide = panel.dates[panel.dates < start][-1]
        w.loc[decide] = 0.0
        w.loc[decide, market.benchmark] = 1.0
        return w, lambda d, t, side: "buy and hold, never trades again", {}

    if kind == "bench_eq":
        w = pd.DataFrame(np.nan, index=panel.dates, columns=panel.tickers)
        decide = panel.dates[panel.dates < start][-1]
        w.loc[decide] = 1.0 / len(panel.tickers)
        return w, lambda d, t, side: "held at equal weight from the first live day, never rebalanced", {}

    if kind == "tournament":
        menu = variants.tournament_variants(panel, market.max_positions, market.cost_rate)
        w, chosen, scores = selftune.run(menu, panel, market.cost_rate, variants.TOUR_FIXED)
        parts = {
            pset: {name: fn(panel, market.max_positions, **params.get(name, {})) for name, fn in STRATEGIES.items()}
            for pset, params in variants.TOUR_PARAMS.items()
        }

        def explain(d, t, side):
            vid = chosen.get(d, variants.TOUR_FIXED)
            pset = "default" if vid == variants.TOUR_FIXED else vid.split(":")[1].split("-")[0]
            holders = [name.replace("_", " ") for name, wts in parts[pset].items() if wts.at[d, t] > 0]
            setting = variants.describe(vid)
            if holders:
                return f"{' and '.join(holders)} hold it under {setting}"
            return f"no strategy holds it any more under {setting}"

        return w, explain, _diag(chosen, scores, start, menu[variants.TOUR_FIXED])

    # ML: the model's odds first, then the menu of rules that turn odds into positions. Shadow history begins
    # WARMUP trading days before `start`, so the first live choice rests on a real record.
    dates = panel.dates
    first = max(0, dates.get_loc(dates[dates < start][-1]) - selftune.WARMUP)
    _, ml_diag = ml.weights(panel, market.max_positions, start=dates[first])
    probs = ml_diag.pop("probs")
    menu = variants.ml_variants(probs, market.max_positions)
    w, chosen, scores = selftune.run(menu, panel, market.cost_rate, variants.ML_FIXED)

    def explain(d, t, side):
        if d not in probs.index:
            return "decided before the model's first scored day"
        p = probs.at[d, t]
        setting = variants.describe(chosen.get(d, variants.ML_FIXED))
        if np.isnan(p):
            return "no model yet (needs a year of training data)"
        if side == "buy":
            return f"model gives {p:.0%} odds of a rise; setting: {setting}"
        return f"model odds now {p:.0%}; sold under the setting: {setting}"

    diag = _diag(chosen, scores, start, menu[variants.ML_FIXED])
    diag.update(ml_diag)
    diag["probs"] = probs  # saved by the backtest for scripts/controls.py; dropped from the summary
    return w, explain, diag


def _diag(chosen: pd.Series, scores: pd.DataFrame, start, fixed_weights: pd.DataFrame) -> dict:
    return {
        "chosen": chosen,
        "scores": scores,
        "fixed_weights": fixed_weights,
        "variant": chosen.iloc[-1],
        "switches": _switches(chosen, start),
    }


def attach_reasons(trades: pd.DataFrame, panel, explain) -> pd.DataFrame:
    """A trade filled on day d was decided at the close of the previous trading day."""
    if trades.empty:
        return trades.assign(reason=pd.Series(dtype=str))
    dates = panel.dates
    prev = {d: dates[i - 1] for i, d in enumerate(dates) if i > 0}
    trades = trades.copy()
    trades["reason"] = [explain(prev[r.date], r.ticker, r.side) for r in trades.itertuples()]
    return trades


__all__ = ["ACCOUNTS", "ML_THRESHOLD", "attach_reasons", "targets"]
