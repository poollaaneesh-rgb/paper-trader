"""ML engine: gradient-boosted trees predicting next-holding-period direction, retrained weekly."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from paper_trader import features as F
from paper_trader.config import ML_THRESHOLD, RETRAIN_EVERY

MIN_TRAIN_DAYS = 252


def boundaries(dates: pd.DatetimeIndex, every: str = RETRAIN_EVERY) -> pd.DatetimeIndex:
    """First trading day of each period."""
    periods = pd.Series(dates.to_period(every), index=dates)
    return dates[(periods != periods.shift(1)).to_numpy()]


def train_end(dates: pd.DatetimeIndex, boundary) -> pd.Timestamp | None:
    """Last row usable for training at `boundary`: its label needs the open two days later."""
    pos = dates.get_loc(boundary) - 2
    return dates[pos] if pos >= 0 else None


def _models():
    hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=15,
                                         random_state=0)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    return hgb, lr


def weights(panel, max_positions: int, start, retrain: str = RETRAIN_EVERY,
            threshold: float = ML_THRESHOLD):
    """Target weights from `start` on, plus diagnostics (retrains, out-of-sample hit rates)."""
    dates = panel.dates
    start = pd.Timestamp(start)
    X = F.build(panel).dropna()
    y = F.labels(panel)
    out = pd.DataFrame(0.0, index=dates[dates >= start], columns=panel.tickers)
    bounds = boundaries(dates, retrain)
    preds = []
    retrains = 0
    for i, b in enumerate(bounds):
        nxt = bounds[i + 1] if i + 1 < len(bounds) else dates[-1] + pd.Timedelta(days=1)
        days = dates[(dates >= max(b, start)) & (dates < nxt)]
        if len(days) == 0:
            continue
        end = train_end(dates, b)
        if end is None:
            continue
        train_dates = X.index.get_level_values("date")
        mask = train_dates <= end
        Xt = X[mask]
        yt = y.reindex(Xt.index)
        keep = yt.notna().to_numpy()
        Xt, yt = Xt[keep], yt[keep]
        if Xt.index.get_level_values("date").nunique() < MIN_TRAIN_DAYS or yt.nunique() < 2:
            continue
        hgb, lr = _models()
        hgb.fit(Xt.to_numpy(), yt.to_numpy())
        lr.fit(Xt.to_numpy(), yt.to_numpy())
        retrains += 1
        Xp = X[np.isin(train_dates, days)]
        if Xp.empty:
            continue
        p = pd.Series(hgb.predict_proba(Xp.to_numpy())[:, 1], index=Xp.index)
        p_lr = pd.Series(lr.predict_proba(Xp.to_numpy())[:, 1], index=Xp.index)
        preds.append(pd.DataFrame({"hgb": p, "lr": p_lr}))
        for d, row in p.groupby(level="date"):
            row = row.droplevel("date")
            picks = row[row > threshold].nlargest(max_positions)
            out.loc[d, picks.index] = 1.0 / max_positions
    diag = {"retrains": retrains}
    if preds:
        allp = pd.concat(preds)
        truth = y.reindex(allp.index)
        known = truth.notna()
        for name in ("hgb", "lr"):
            hit = ((allp[name] > 0.5).astype(float) == truth)[known]
            diag[f"{name}_hit_rate"] = round(float(hit.mean()), 4) if len(hit) else None
        diag["predictions_scored"] = int(known.sum())
    return out, diag
