"""Decision records: enough to recompute any night's targets exactly (results/live/decisions.csv).

Every engine is a pure function of the price panel: the targets decided at close d come from the frozen prices
through that night's last close and the code that ran. So each decision records the data date (the panel's last
close that night), a digest of that panel, the code version (git short hash), the setting in force and the targets
themselves. `python -m paper_trader.replay` rebuilds the panel, checks it against the digest and reruns that code in a
sandbox. The file is append-only like the rest of the live record and is never recomputed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ("date", "account", "data_date", "data_digest", "code", "setting", "following", "targets")


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


def row(d, name: str, panel, panel_digest: str, a: dict) -> dict:
    """The record of the targets account `a` holds pending after the close of `d`."""
    following = a.get("following")
    return {
        "date": pd.Timestamp(d),
        "account": name,
        "data_date": str(panel.dates.max().date()),
        "data_digest": panel_digest,
        "code": "",  # filled in by run_daily, which knows the checkout
        "setting": a.get("variant") or "",
        "following": json.dumps(following["variant"], sort_keys=True) if following else "",
        "targets": targets_text(a.get("pending")),
    }


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(FIELDS))
    return pd.read_csv(path, dtype=str, keep_default_na=False)
