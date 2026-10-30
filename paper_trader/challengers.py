"""Challengers written by the nightly review: run forward only, and adopted only past a bar.

Since 2026-10-29 every bot's trades are reviewed each night. A defect is fixed in code with a test; a change of
strategy is not made in the account but entered in results/live/challengers.json as a challenger. The menu layer
(selftune.py) scores its entries on the trailing WINDOW, which is fair to a menu written before any result was seen
but would score a setting written after seeing the losses on those same days. So a challenger has no past. It starts
from a copy of its account's book at the close of the day it was added (`started`), trades its own targets on the same
prices and costs from the next day, and is judged on those days only. Nothing before them is ever computed for it.
Its state is kept in state.json beside the accounts it shadows, and its daily equity in results/live/challengers.csv.

The bar is the menu layer's. Once it has MIN_DAYS trading days of its own, a challenger whose equity is ahead of the
live account's by MARGIN (log return over exactly its own days; both start from the same book) takes over the account
through the settings log. One behind by MARGIN after MIN_DAYS, or still short of the bar after MAX_DAYS, is retired.
After a takeover the account follows the challenger and the menu layer's choice is no longer applied to it.

An entry, one per challenger (at most MAX_ACTIVE active per account):

    {"id": "ml-stocks-hold3", "account": "ml_stocks", "variant": {"entry": 0.57, "gap": 0.05, "hold": 3},
     "hypothesis": "...", "rule_id": "...", "started": "2026-10-29", "status": "active"}

`variant` takes the parameters the menus use (variants.check_definition). An entry that sets cash, costs, sizing or
the universe is refused and the reason logged; those belong to the owner, not to a review. The nightly run fills in
`status`, `ended` and `result` when a challenger is promoted or retired. `python -m paper_trader.challengers` checks
the file against the live state before it is committed.
"""

from __future__ import annotations

import copy
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from paper_trader import engines, selftune, simulate, variants
from paper_trader.portfolio import Account

MIN_DAYS = 30  # trading days of its own before a challenger can take over, or be retired for trailing
MAX_DAYS = 120  # trading days of its own after which a challenger that has not cleared the bar is retired
MARGIN = selftune.MARGIN  # the lead needed to take over, in log return over the challenger's own days
MAX_ACTIVE = 2  # active challengers per account
STATUSES = ("active", "promoted", "retired")
FIELDS = ("id", "account", "variant", "hypothesis", "rule_id", "started", "status")
WRITTEN = ("ended", "result")  # filled in by the nightly run
ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
# A key containing any of these words, anywhere in an entry, sets risk or capital and refuses the entry.
RISK_WORDS = (
    "cash",
    "capital",
    "fee",
    "cost",
    "slippage",
    "universe",
    "ticker",
    "position",
    "size",
    "leverage",
    "weight",
    "benchmark",
    "gate",
    "bankrupt",
)


def risk_keys(entry) -> list[str]:
    """Keys anywhere in an entry that name a risk or capital parameter."""
    found = []
    if isinstance(entry, dict):
        for k, v in entry.items():
            if any(word in str(k).lower() for word in RISK_WORDS):
                found.append(str(k))
            found += risk_keys(v)
    elif isinstance(entry, list):
        for v in entry:
            found += risk_keys(v)
    return found


def verdict(days: int, lead: float) -> str | None:
    """'promote', 'retire' or None for a challenger with `days` trading days of its own and this lead."""
    if days < MIN_DAYS:
        return None
    if lead >= MARGIN:
        return "promote"
    if lead <= -MARGIN or days >= MAX_DAYS:
        return "retire"
    return None


def load(path: Path) -> list[dict]:
    """The challengers file; a missing file is an empty list. Raises ValueError if it is not a JSON list."""
    if not path.exists():
        return []
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError("challengers.json must hold a list of entries")
    return entries


def save(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps(entries, indent=2) + "\n")


def _pending(w: pd.DataFrame, d) -> dict | None:
    row = w.loc[d] if d in w.index else None
    return None if row is None or row.isna().all() else row.fillna(0.0).to_dict()


def _series(d: dict | None):
    return None if d is None else pd.Series(d, dtype=float)


def _lead(equity: float, live_equity: float) -> float:
    """Log return of the challenger less the live account's over the same days; both started from the same book."""
    return math.log(max(equity, 1e-9) / max(live_equity, 1e-9))


@dataclass
class Runner:
    """An active challenger while the run steps its account."""

    id: str
    st: dict  # its entry in state["challengers"]
    weights: pd.DataFrame
    explain: Callable
    acct: Account
    avg_cost: dict


class Book:
    """The file's entries for one nightly run. live.run steps them; run_daily writes `rows` and prints `events`."""

    def __init__(self, entries: list[dict]):
        self.entries = entries
        self.rows: list[dict] = []  # results/live/challengers.csv
        self.events: list[str] = []
        self._admitted: dict[str, list[dict]] = {}

    # ---- the file ----
    def admit(self, state: dict, accounts: dict, today) -> None:
        """Check each entry not yet started: refuse it with a logged reason, or queue it to start with its account."""
        known = state.setdefault("challengers", {})
        today = pd.Timestamp(today).normalize()
        active = Counter(st["account"] for st in known.values() if st["status"] == "active")
        seen: set[str] = set()
        self._admitted = {}
        for entry in self.entries:
            cid = entry.get("id") if isinstance(entry, dict) else None
            if isinstance(cid, str) and cid in known:
                self._follow(state, entry, known[cid], today)
                seen.add(cid)
                continue
            reason = self._refusal(entry, state, accounts, today, seen, active)
            if isinstance(cid, str):
                seen.add(cid)
            if reason:
                self._refuse(state, cid, reason, today)
                continue
            active[entry["account"]] += 1
            self._admitted.setdefault(entry["account"], []).append(entry)
        refused = state.get("challengers_refused", {})
        for key in [k for k in refused if k not in seen and k != "(no id)"]:
            del refused[key]  # the review has taken the entry out of the file

    def _refusal(self, entry, state, accounts, today, seen, active) -> str | None:
        if not isinstance(entry, dict):
            return "an entry must be a table of fields"
        risk = risk_keys(entry)
        if risk:
            return (
                f"sets risk or capital parameters ({', '.join(risk)}); a challenger may change the engine's setting "
                "only, never cash, costs, sizing or the universe"
            )
        missing = [f for f in FIELDS if entry.get(f) in (None, "")]
        if missing:
            return f"missing {', '.join(missing)}"
        extra = sorted(set(entry) - set(FIELDS) - set(WRITTEN))
        if extra:
            return f"unknown fields {', '.join(extra)}"
        cid, name = entry["id"], entry["account"]
        if not (isinstance(cid, str) and ID.fullmatch(cid)):
            return "id must be lower-case letters, digits, '.', '_' or '-'"
        if cid in seen:
            return "the same id appears twice in the file"
        if entry["status"] != "active":
            return f"status is {entry['status']!r}; a new challenger starts as 'active'"
        kind = accounts.get(name, (None, None))[1]
        if kind not in ("tournament", "ml"):
            return f"{name} is not a tournament or ML account"
        bad = variants.check_definition(kind, entry["variant"])
        if bad:
            return bad
        if not (isinstance(entry["started"], str) and DATE.fullmatch(entry["started"])):
            return "started must be a date written YYYY-MM-DD"
        started = pd.Timestamp(entry["started"])
        if started > today:
            return f"started {entry['started']} is after today ({today.date()}); write the day it is added"
        a = state.get("accounts", {}).get(name)
        if a and a.get("bankrupt"):
            return f"{name} is bankrupt and trades no more"
        if a and a.get("initialised") and started < pd.Timestamp(a["last_date"]):
            return (
                f"started {entry['started']} is before {name}'s last recorded day {a['last_date']}, so its record "
                "would be backfilled; a challenger starts on the day it is added"
            )
        if active[name] >= MAX_ACTIVE:
            return f"{name} already has {MAX_ACTIVE} active challengers"
        return None

    def _refuse(self, state, cid, reason, today) -> None:
        key = cid if isinstance(cid, str) else "(no id)"
        refused = state.setdefault("challengers_refused", {})
        if refused.get(key, {}).get("reason") != reason:
            refused[key] = {"run": str(today.date()), "reason": reason}
        self.events.append(f"challenger {key} refused: {reason}")

    def _follow(self, state, entry, st, today) -> None:
        """A started challenger runs as it started; the review may withdraw it, but not change or promote it."""
        if not isinstance(entry, dict):
            return
        if entry.get("status") == "retired" and st["status"] == "active":
            _end(st, "retired", today, "withdrawn by the review")
            self.events.append(f"challenger {entry['id']} withdrawn by the review")
            return
        if any(entry.get(f) != st[f] for f in ("account", "variant", "started")):
            note = "changed after it started; the setting it started with keeps running"
            self._refuse(state, entry["id"], note, today)
        else:
            state.get("challengers_refused", {}).pop(entry["id"], None)

    def sync(self, state: dict) -> bool:
        """Copy status, end date and result from the state into the file's entries. True if anything changed."""
        known = state.get("challengers", {})
        changed = False
        for entry in self.entries:
            st = known.get(entry.get("id")) if isinstance(entry, dict) else None
            if st is None:
                continue
            for f in ("status", *WRITTEN):
                if st.get(f) is not None and entry.get(f) != st[f]:
                    entry[f] = st[f]
                    changed = True
        return changed

    # ---- stepping, called by live.run for each account with new days ----
    def open(self, state, name, kind, market, panel, diag, a, acct, avg_cost) -> list[Runner]:
        """Start the account's queued challengers from a copy of its book, and return every active one."""
        known = state.setdefault("challengers", {})
        new = set()
        for entry in self._admitted.pop(name, []):
            known[entry["id"]] = {
                "account": name,
                "variant": copy.deepcopy(entry["variant"]),
                "description": variants.describe_definition(kind, entry["variant"]),
                "hypothesis": entry["hypothesis"],
                "rule_id": entry["rule_id"],
                "started": entry["started"],
                "status": "active",
                "days": 0,
                "equity": None,
                "lead": None,
                "ended": None,
                "result": None,
                "last_date": a["last_date"],
                "cash": acct.cash,
                "positions": dict(acct.positions),
                "avg_cost": dict(avg_cost),
                "pending": None,
            }
            state.setdefault("challengers_refused", {}).pop(entry["id"], None)
            new.add(entry["id"])
            self.events.append(
                f"challenger {entry['id']} started on {name}; its record begins after {entry['started']}"
            )
        runners = []
        for cid, st in known.items():
            if st["account"] != name or st["status"] != "active":
                continue
            w, explain = engines.challenger_targets(kind, market, panel, st["variant"], diag)
            if cid in new:
                st["pending"] = _pending(w, pd.Timestamp(st["last_date"]))
            acct_copy = Account(cash=st["cash"], positions=dict(st["positions"]))
            runners.append(Runner(cid, st, w, explain, acct_copy, dict(st["avg_cost"])))
        return runners

    def day(self, runners, d, acct, avg_cost, live_equity, panel, cost_rate) -> Runner | None:
        """Step each challenger through day `d`; return the one to promote, if any cleared the bar."""
        best = None
        for r in list(runners):
            st = r.st
            if d <= pd.Timestamp(st["last_date"]):
                continue
            own = d > pd.Timestamp(st["started"])
            if own:
                fills = simulate.step(r.acct, d, panel, _series(st["pending"]), cost_rate, r.avg_cost)
                equity = round(simulate.mark(r.acct, d, panel), 2)
                st["days"] += 1
                st["equity"], st["lead"] = equity, _lead(equity, live_equity)
                self.rows.append(
                    {"date": d, "account": st["account"], "challenger": r.id, "equity": equity, "trades": len(fills)}
                )
            else:  # the day it was added or before: no record yet, only a copy of the live book
                r.acct, r.avg_cost = Account(cash=acct.cash, positions=dict(acct.positions)), dict(avg_cost)
            st["pending"] = _pending(r.weights, d)
            st["last_date"] = d.date().isoformat()
            st["cash"], st["positions"], st["avg_cost"] = r.acct.cash, dict(r.acct.positions), dict(r.avg_cost)
            if not own:
                continue
            v = verdict(st["days"], st["lead"])
            if v == "retire":
                if st["lead"] <= -MARGIN:
                    why = f"behind the live setting by {-st['lead'] * 100:.1f}% after {st['days']} days"
                else:
                    why = f"{st['lead'] * 100:+.1f}% against the live setting after {st['days']} days, short of the bar"
                _end(st, "retired", d, why)
                self.events.append(f"challenger {r.id} retired: {st['result']}")
                runners.remove(r)
            elif v == "promote" and (best is None or st["lead"] > best.st["lead"]):
                best = r
        return best

    def promote(self, state, runners, r: Runner, a: dict, name: str, d, live_equity: float) -> dict:
        """Hand the account to challenger `r`; return its settings row."""
        st = r.st
        prev = (a.get("following") or {}).get("id")
        if prev in state.get("challengers", {}):
            _end(state["challengers"][prev], "retired", d, f"replaced by {r.id}")
        _end(st, "promoted", d, f"ahead of the live setting by {st['lead'] * 100:.1f}% over {st['days']} days")
        runners.remove(r)
        a["following"] = {"id": r.id, "variant": copy.deepcopy(st["variant"])}
        a["variant"] = f"review:{r.id}"
        self.events.append(f"challenger {r.id} promoted on {name}: {st['result']}")
        reason = (
            f"review challenger {r.id} (rule {st['rule_id']}) ahead of the live setting by {st['lead'] * 100:.1f}% "
            f"over its {st['days']} trading days since {st['started']} (${st['equity']:,.2f} against "
            f"${live_equity:,.2f}), after costs; hypothesis: {st['hypothesis']}"
        )
        return {"date": d, "account": name, "variant": a["variant"], "description": st["description"], "reason": reason}

    def close_all(self, runners, d, reason: str) -> None:
        for r in list(runners):
            _end(r.st, "retired", d, reason)
            self.events.append(f"challenger {r.id} retired: {reason}")
            runners.remove(r)


def _end(st: dict, status: str, d, result: str) -> None:
    st["status"], st["ended"], st["result"] = status, str(pd.Timestamp(d).date()), result


def summary(state: dict) -> dict:
    """Every challenger's record so far, and the entries refused, for summary.json."""
    keep = ("account", "description", "rule_id", "started", "status", "days", "equity", "lead", "ended", "result")
    return {
        "challengers": {cid: {k: st.get(k) for k in keep} for cid, st in state.get("challengers", {}).items()},
        "refused": state.get("challengers_refused", {}),
    }


def main(root: Path) -> int:
    """Check results/live/challengers.json against the live state, as tonight's run would; print what it would do."""
    live_dir = root / "results" / "live"
    state_path = live_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"accounts": {}}
    book = Book(load(live_dir / "challengers.json"))
    book.admit(state, engines.ACCOUNTS, pd.Timestamp.now(tz="UTC").tz_localize(None))
    for name, entries in book._admitted.items():
        for entry in entries:
            print(f"challenger {entry['id']} ok: starts on {name} at the next run")
    for line in book.events:
        print(line)
    return 1 if any(" refused: " in line for line in book.events) else 0


if __name__ == "__main__":
    sys.exit(main(Path(__file__).resolve().parents[1]))
