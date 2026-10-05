"""The menus of settings each engine can move between, and plain-English names for them.

Each variant is a complete table of target weights, so the self-tuning layer can score it as a shadow account.
The menus were written down once; they do not grow with results.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from paper_trader import tournament
from paper_trader.strategies import STRATEGIES

# ---- ML: how the model's odds turn into positions ----
ML_FIXED = "ml:rerank55"  # the original rule: the top names above 55% odds, re-ranked every day
ML_CASH = "ml:cash"  # sit in cash: the honest answer when nothing in the menu is paying for its costs
ML_ENTRY = (0.52, 0.55, 0.58, 0.62)
ML_GAP = (0.0, 0.03, 0.07)  # exit line = entry line minus the gap
ML_HOLD = (1, 5, 10)  # minimum days held


def ml_rerank(probs: pd.DataFrame, max_positions: int, threshold: float) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=probs.index, columns=probs.columns)
    for d, row in probs.iterrows():
        picks = row[row > threshold].nlargest(max_positions)
        out.loc[d, picks.index] = 1.0 / max_positions
    return out


def ml_band(probs: pd.DataFrame, max_positions: int, th_in: float, th_out: float, min_hold: int) -> pd.DataFrame:
    """Enter a name when its odds pass th_in; keep it until the odds fall below th_out and min_hold days have passed."""
    values = probs.to_numpy()
    out = np.zeros(values.shape)
    held: dict[int, int] = {}  # column -> days held so far
    for i, p in enumerate(values):
        for j in list(held):
            if held[j] >= min_hold and (np.isnan(p[j]) or p[j] < th_out):
                del held[j]
        free = max_positions - len(held)
        if free > 0:
            candidates = [(p[j], j) for j in range(len(p)) if j not in held and not np.isnan(p[j]) and p[j] > th_in]
            for _, j in sorted(candidates, reverse=True)[:free]:
                held[j] = 0
        for j in held:
            held[j] += 1
            out[i, j] = 1.0 / max_positions
    return pd.DataFrame(out, index=probs.index, columns=probs.columns)


def _ml_id(th_in: float, gap: float, hold: int) -> str:
    return f"ml:in{round(th_in * 100)}-out{round((th_in - gap) * 100)}-hold{hold}"


def ml_variants(probs: pd.DataFrame, max_positions: int) -> dict[str, pd.DataFrame]:
    menu = {ML_FIXED: ml_rerank(probs, max_positions, 0.55)}
    for th_in, gap, hold in itertools.product(ML_ENTRY, ML_GAP, ML_HOLD):
        menu[_ml_id(th_in, gap, hold)] = ml_band(probs, max_positions, th_in, th_in - gap, hold)
    menu[ML_CASH] = pd.DataFrame(0.0, index=probs.index, columns=probs.columns)
    return menu


# ---- Tournament: strategy speeds, lookback, cadence and scoring ----
TOUR_FIXED = "tour:default-63-M-gross"
TOUR_PARAMS: dict[str, dict[str, dict]] = {
    "default": {},
    "slow": {
        "momentum": {"lookback": 126},
        "mean_reversion": {"buy_below": 25, "sell_above": 55},
        "trend": {"fast": 100},
    },
    "fast": {
        "momentum": {"lookback": 21},
        "mean_reversion": {"rsi_n": 2, "buy_below": 10, "sell_above": 60},
        "trend": {"fast": 20, "slow": 50},
    },
}
TOUR_LOOKBACKS = (21, 63, 126, 252)
TOUR_CADENCES = ("M", "W", "D")


def tournament_variants(panel, max_positions: int, cost_rate: float) -> dict[str, pd.DataFrame]:
    menu = {TOUR_FIXED: tournament.weights(panel, max_positions)[0]}
    for pset, params in TOUR_PARAMS.items():
        per = {name: fn(panel, max_positions, **params.get(name, {})) for name, fn in STRATEGIES.items()}
        rets = pd.DataFrame({name: tournament.strategy_returns(w, panel, cost_rate) for name, w in per.items()})
        for lookback, cadence in itertools.product(TOUR_LOOKBACKS, TOUR_CADENCES):
            alloc = tournament.allocations(rets, lookback, cadence)
            menu[f"tour:{pset}-{lookback}-{cadence}-net"] = sum(per[n].mul(alloc[n], axis=0) for n in per)
    return menu


# ---- names ----
_CADENCE = {"M": "monthly", "W": "weekly", "D": "daily"}
_SPEED = {"default": "the original strategies", "slow": "slower strategies", "fast": "faster strategies"}


def describe(variant_id: str) -> str:
    """Plain English for a variant id; an id the menus never issued is returned as it is."""
    try:
        return _describe(variant_id)
    except (ValueError, KeyError):
        return variant_id


def _describe(variant_id: str) -> str:
    kind, _, rest = variant_id.partition(":")
    if variant_id == ML_FIXED:
        return "the original rule: the top names above 55% odds, re-ranked every day"
    if variant_id == ML_CASH:
        return "sit in cash: nothing in the menu is paying for its costs"
    if variant_id == TOUR_FIXED:
        return "the original rules: capital re-weighted monthly by the last 63 days before costs"
    if kind == "ml":
        th_in, th_out, hold = rest.split("-")
        return f"enter above {th_in[2:]}% odds, exit below {th_out[3:]}%, hold at least {hold[4:]} days"
    if kind == "tour":
        pset, lookback, cadence, scoring = rest.split("-")
        return (
            f"{_SPEED[pset]}, capital re-weighted {_CADENCE[cadence]} by the last {lookback} days "
            f"{'after' if scoring == 'net' else 'before'} costs"
        )
    return variant_id
