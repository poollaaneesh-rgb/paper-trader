"""Controls for the backtest: the bar the self-tuned results should be held to.

1. Holding every name in the universe at equal weight (the universe is today's winners, so this bar is high).
2. A no-skill control for the ML accounts: the model's odds shuffled across names each day, then the same menu and
   the same self-tuning layer. Run after scripts/backtest.py; writes results/backtest/controls.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_trader import selftune, variants
from paper_trader.config import BACKTEST_END, BACKTEST_START, MARKETS, START_CASH
from scripts.backtest import OUT, frozen_panel

SHUFFLES = 5


def final_value(returns: pd.Series) -> float:
    return float(START_CASH * np.prod(1 + returns.loc[BACKTEST_START:BACKTEST_END]))


def equal_weight(panel, cost_rate: float) -> float:
    """Every name with a price, equal weight, rebalanced daily, costs charged."""
    avail = panel.close.notna().astype(float)
    w = avail.div(avail.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return final_value(selftune.net_returns(w, panel, cost_rate))


def equal_weight_hold(panel) -> float:
    """Buy every name that had a price on the first backtest day, equal amounts, and never trade again."""
    opens = panel.open.loc[BACKTEST_START:BACKTEST_END]
    first = opens.iloc[0].dropna()
    last = panel.close.loc[:BACKTEST_END].ffill().iloc[-1]
    return float(START_CASH * (last[first.index] / first).mean())


def shuffled(panel, probs: pd.DataFrame, max_positions: int, cost_rate: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    values = probs.to_numpy().copy()
    for row in values:
        ok = ~np.isnan(row)
        row[ok] = rng.permutation(row[ok])
    menu = variants.ml_variants(pd.DataFrame(values, index=probs.index, columns=probs.columns), max_positions)
    w, _, _ = selftune.run(menu, panel, cost_rate, variants.ML_FIXED)
    return final_value(selftune.net_returns(w, panel, cost_rate))


def main():
    out = {}
    for key, market in MARKETS.items():
        panel = frozen_panel(key, market)
        probs = pd.read_csv(OUT / f"ml_probs_{key}.csv.gz", index_col=0, parse_dates=True)
        runs = [
            round(shuffled(panel, probs, market.max_positions, market.cost_rate, s), 2) for s in range(1, SHUFFLES + 1)
        ]
        out[key] = {
            "equal_weight_hold": round(equal_weight_hold(panel), 2),
            "equal_weight_daily": round(equal_weight(panel, market.cost_rate), 2),
            "shuffled_ml_selftuned": runs,
            "shuffled_ml_median": round(float(np.median(runs)), 2),
        }
        print(key, out[key])
    (OUT / "controls.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
