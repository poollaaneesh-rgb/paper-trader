"""Decision records: enough to recompute any night's targets exactly (results/live/decisions.csv).

Every engine is a pure function of the price panel: the targets decided at close d come from the frozen prices
through that night's last close and the code that ran. So each decision records the data date (the panel's last
close that night), a digest of that panel, the code version (git short hash), the setting in force and the targets
themselves. `python -m paper_trader.replay` rebuilds the panel, checks it against the digest and reruns that code in a
sandbox. The file is append-only like the rest of the live record and is never recomputed.

Since 2026-11-12 each row also carries the fields every book in the trading system records (the decision record of
the revamp, step 4): its own id, the team and book, the wall-clock time the run decided (a second time beside the
data date), where the signal came from, the model's setting when a model made it, every name the account could have
held that night with the model's odds where there are any (so names it passed over are on record too), and the limits
in force. Three of the shared fields have no column here and the reason is the same each night: an account fills at
the next open from daily bars, so there is no bid, ask or size to record; the fee is the market's flat rate (in
`limits`) and what was paid is in trades.csv; and the outcome is equity.csv and trades.csv, which only this code
writes. Rows written before that date keep their eight values and have the new columns blank.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

BASE_FIELDS = ("date", "account", "data_date", "data_digest", "code", "setting", "following", "targets")
RECORD_FIELDS = ("id", "team", "book", "decided_at", "signal_source", "model_id", "candidates", "limits")
FIELDS = BASE_FIELDS + RECORD_FIELDS

# Where each kind of account's targets come from. A benchmark holds its market by rule.
SIGNAL_SOURCE = {"tournament": "rule", "ml": "model", "bench": "rule", "bench_eq": "rule"}


def code_version(root: Path) -> str:
    """The short hash of the code that runs tonight, or 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=10", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = os.environ.get("GITHUB_SHA", "")
        return sha[:10] if sha else "unknown"


def digest(panel) -> str:
    """A short fingerprint of a price panel: tickers, dates and every open, close and volume."""
    h = hashlib.sha1()
    h.update(json.dumps(list(panel.tickers)).encode())
    h.update(np.asarray(panel.dates.asi8, dtype=np.int64).tobytes())
    for frame in (panel.open, panel.close, panel.volume):
        h.update(np.ascontiguousarray(frame.to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]


def targets_text(pending: dict | None) -> str:
    """Targets as 'TICKER=weight;...' (sorted, 6 places, zero weights left out); '' means hold, 'cash' means none."""
    if pending is None:
        return ""
    parts = [f"{t}={float(w):.6f}" for t, w in sorted(pending.items()) if w and abs(float(w)) > 5e-7]
    return ";".join(parts) or "cash"


def parse_targets(text) -> dict | None:
    """The inverse of targets_text: None for hold, {} for all cash."""
    if text is None or (isinstance(text, float) and np.isnan(text)) or text == "":
        return None
    if text == "cash":
        return {}
    return {t: float(w) for t, w in (part.split("=") for part in str(text).split(";"))}


def same_targets(a: dict | None, b: dict | None, tol: float = 2e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return all(abs(a.get(t, 0.0) - b.get(t, 0.0)) <= tol for t in set(a) | set(b))


def team_of(market_name: str, kind: str) -> str:
    """The team a nightly account belongs to: the one crypto team, or its own stock strategy team."""
    return "crypto" if market_name == "crypto" else f"stocks:{kind}"


def candidates_text(d, panel, probs=None) -> str:
    """Every name the account could hold after the close of `d`, as 'TICKER=odds;...' (sorted).

    A name is a candidate when it has a close on `d`. The odds are the model's for that close (4 places); they are
    left empty for an account whose rule gives no per-name number, and for a name the model did not score.
    """
    day = pd.Timestamp(d)
    if day not in panel.close.index:
        return ""
    closes = panel.close.loc[day]
    odds = probs.loc[day] if probs is not None and day in probs.index else None
    parts = []
    for t in sorted(closes.index[closes.notna()]):
        p = None if odds is None else odds.get(t)
        parts.append(f"{t}={float(p):.4f}" if p is not None and not pd.isna(p) else f"{t}=")
    return ";".join(parts)


def row(d, name: str, panel, panel_digest: str, a: dict, kind: str = "", market=None, probs=None) -> dict:
    """The record of the targets account `a` holds pending after the close of `d`."""
    following = a.get("following")
    setting = a.get("variant") or ""
    limits = {"bankrupt": bool(a.get("bankrupt")), "on_watch": bool(a.get("watch"))}
    if market is not None:
        limits = {"cost_rate": market.cost_rate, "max_positions": market.max_positions, **limits}
    record = {
        "id": f"{name}|{pd.Timestamp(d).date()}",
        "team": team_of(market.name, kind) if market is not None else "",
        "book": "rules",
        "decided_at": "",  # filled in by run_daily, which knows the time of the run
        "signal_source": SIGNAL_SOURCE.get(kind, ""),
        "model_id": (json.dumps(following["variant"], sort_keys=True) if following else setting)
        if kind == "ml"
        else "",
        "candidates": candidates_text(d, panel, probs if kind == "ml" else None),
        "limits": json.dumps(limits, sort_keys=True),
    }
    return {
        "date": pd.Timestamp(d),
        "account": name,
        "data_date": str(panel.dates.max().date()),
        "data_digest": panel_digest,
        "code": "",  # filled in by run_daily, which knows the checkout
        "setting": setting,
        "following": json.dumps(following["variant"], sort_keys=True) if following else "",
        "targets": targets_text(a.get("pending")),
        **record,
    }


def widen(path: Path) -> bool:
    """Give a decisions file written before the record fields the new columns, blank. Returns whether it changed.

    Appending a wider row under a narrower header would make the file unreadable, so the header is widened first.
    No value already in the file changes: the old columns are written back exactly as they were read.
    """
    if not path.exists():
        return False
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in FIELDS if c not in df.columns]
    if not missing:
        return False
    for c in missing:
        df[c] = ""
    df.to_csv(path, index=False)
    return True


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(FIELDS))
    return pd.read_csv(path, dtype=str, keep_default_na=False)
