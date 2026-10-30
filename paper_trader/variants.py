"""The menus of settings each engine can move between, and plain-English names for them.

Each variant is a complete table of target weights, so the self-tuning layer can score it as a shadow account.
The menus were written down once; they do not grow with results.
"""

from __future__ import annotations

import inspect
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


# ---- one setting written by a nightly review (challengers.py), in the menus' own parameters ----
# ML: {"rule": "band", "entry": 0.57, "gap": 0.05, "hold": 3} (rule defaults to band), {"rule": "rerank",
# "threshold": 0.6} or {"rule": "cash"}. Tournament: {"speeds": "slow", "params": {"momentum": {"lookback": 84}},
# "lookback": 42, "cadence": "W", "scoring": "net"}; speeds defaults to default, params to none, scoring to net.
ML_RULE_KEYS = {"band": {"rule", "entry", "gap", "hold"}, "rerank": {"rule", "threshold"}, "cash": {"rule"}}
TOUR_KEYS = {"speeds", "params", "lookback", "cadence", "scoring"}
TOUR_SCORING = ("net", "gross")


def _number(x) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and np.isfinite(x)


def _strategy_args(name: str) -> set[str]:
    return set(inspect.signature(STRATEGIES[name]).parameters) - {"panel", "max_positions"}


def check_definition(kind: str, d) -> str | None:
    """Why a written setting cannot run, or None. Only the parameters the menus vary are accepted."""
    if not isinstance(d, dict):
        return "the variant must be a table of settings"
    if kind == "ml":
        return _check_ml(d)
    if kind == "tournament":
        return _check_tournament(d)
    return f"the {kind} engine has no settings to change"


def _check_ml(d: dict) -> str | None:
    rule = d.get("rule", "band")
    if rule not in ML_RULE_KEYS:
        return f"unknown ML rule {rule!r}; the rules are band, rerank and cash"
    extra = sorted(set(d) - ML_RULE_KEYS[rule])
    if extra:
        return f"the ML {rule} rule has no setting {', '.join(extra)}"
    missing = sorted(ML_RULE_KEYS[rule] - {"rule"} - set(d))
    if missing:
        return f"the ML {rule} rule needs {', '.join(missing)}"
    if rule == "rerank" and not (_number(d["threshold"]) and 0.5 <= d["threshold"] < 1):
        return "threshold must be odds from 0.5 to under 1"
    if rule == "band":
        entry, gap, hold = d["entry"], d["gap"], d["hold"]
        if not (_number(entry) and 0.5 <= entry < 1):
            return "entry must be odds from 0.5 to under 1"
        if not (_number(gap) and 0 <= gap < entry):
            return "gap must be at least 0 and leave an exit line above 0"
        if not (isinstance(hold, int) and not isinstance(hold, bool) and 1 <= hold <= 63):
            return "hold must be a whole number of days from 1 to 63"
    return None


def _check_tournament(d: dict) -> str | None:
    extra = sorted(set(d) - TOUR_KEYS)
    if extra:
        return f"the tournament has no setting {', '.join(extra)}"
    if d.get("speeds", "default") not in TOUR_PARAMS:
        return f"speeds must be one of {', '.join(TOUR_PARAMS)}"
    lookback, cadence = d.get("lookback"), d.get("cadence")
    if not (isinstance(lookback, int) and not isinstance(lookback, bool) and 5 <= lookback <= 504):
        return "lookback must be a whole number of trading days from 5 to 504"
    if cadence not in TOUR_CADENCES:
        return f"cadence must be one of {', '.join(TOUR_CADENCES)}"
    if d.get("scoring", "net") not in TOUR_SCORING:
        return "scoring must be net or gross"
    params = d.get("params", {})
    if not isinstance(params, dict):
        return "params must be a table per strategy"
    for name, kwargs in params.items():
        if name not in STRATEGIES:
            return f"no strategy called {name!r}; the strategies are {', '.join(STRATEGIES)}"
        if not isinstance(kwargs, dict):
            return f"params for {name} must be a table"
        extra = sorted(set(kwargs) - _strategy_args(name))
        if extra:
            return f"{name} has no setting {', '.join(extra)}"
        if not all(_number(v) and v > 0 for v in kwargs.values()):
            return f"settings for {name} must be positive numbers"
    return None


def ml_definition(probs: pd.DataFrame, max_positions: int, d: dict) -> pd.DataFrame:
    """Target weights of a written ML setting; a band or rerank setting matches the menu entry with the same numbers."""
    rule = d.get("rule", "band")
    if rule == "cash":
        return pd.DataFrame(0.0, index=probs.index, columns=probs.columns)
    if rule == "rerank":
        return ml_rerank(probs, max_positions, d["threshold"])
    return ml_band(probs, max_positions, d["entry"], d["entry"] - d["gap"], d["hold"])


def tournament_definition(panel, max_positions: int, cost_rate: float, d: dict):
    """Return (target weights, weights per strategy) of a written tournament setting."""
    params = {name: dict(kwargs) for name, kwargs in TOUR_PARAMS[d.get("speeds", "default")].items()}
    for name, kwargs in d.get("params", {}).items():
        params.setdefault(name, {}).update(kwargs)
    per = {name: fn(panel, max_positions, **params.get(name, {})) for name, fn in STRATEGIES.items()}
    rate = cost_rate if d.get("scoring", "net") == "net" else 0.0
    rets = pd.DataFrame({name: tournament.strategy_returns(w, panel, rate) for name, w in per.items()})
    alloc = tournament.allocations(rets, d["lookback"], d["cadence"])
    return sum(per[n].mul(alloc[n], axis=0) for n in per), per


def describe_definition(kind: str, d: dict) -> str:
    """Plain English for a written setting, in the words describe() uses for the menus."""
    if kind == "ml":
        rule = d.get("rule", "band")
        if rule == "cash":
            return "sit in cash"
        if rule == "rerank":
            return f"the top names above {d['threshold']:.0%} odds, re-ranked every day"
        return _describe(_ml_id(d["entry"], d["gap"], d["hold"]))
    text = _describe(f"tour:{d.get('speeds', 'default')}-{d['lookback']}-{d['cadence']}-{d.get('scoring', 'net')}")
    changes = [
        f"{n.replace('_', ' ')} {k.replace('_', ' ')} {v}"
        for n, kw in d.get("params", {}).items()
        for k, v in kw.items()
    ]
    return f"{text}; with {', '.join(changes)}" if changes else text
