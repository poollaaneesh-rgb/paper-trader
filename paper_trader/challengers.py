"""Challengers written by the nightly review: run forward only, and adopted only past a bar.

Since 2026-10-29 every bot's trades are reviewed each night. A defect is fixed in code with a test; a change of
strategy is not made in the account but entered in results/live/challengers.json as a challenger. The menu layer
(selftune.py) scores its entries on the trailing WINDOW, which is fair to a menu written before any result was seen
but would score a setting written after seeing the losses on those same days. So a challenger has no past. It starts
from a copy of its account's book at the close of the day it was added (`started`), trades its own targets on the same
prices and costs from the next day, and is judged on those days only. Nothing before them is ever computed for it.
Its state is kept in state.json beside the accounts it shadows, and its daily equity in results/live/challengers.csv.

The bar is the owner's decision of 2026-11-05 (the loss replay rule, which replaced the menu layer's 2% after 30
days). Once it has MIN_DAYS trading days of its own, a challenger takes over the account through the settings log when
its return per dollar over exactly those days leads the live account's by more than LEAD (as a share of the live
result; both start from the same book) and a bootstrap of its daily return gaps, with a fixed seed, gives at least
CONFIDENCE that it is better. One behind by LEAD after MIN_DAYS, or still short of the bar after MAX_DAYS, is retired.
After a takeover the account follows the challenger and the menu layer's choice is no longer applied to it.

An adopted challenger is watched: the version it replaced keeps running as a shadow from the same book, and if the
adopted one trails it by REVERT_GAP over its next REVERT_DAYS days, the account goes back to it through the settings
log. One change at a time per account: no challenger takes over while a watch is open or on the day of a revert.

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
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from paper_trader import engines, simulate, variants
from paper_trader.portfolio import Account

MIN_DAYS = 20  # trading days of its own before a challenger can take over, or be retired for trailing
MAX_DAYS = 120  # trading days of its own after which a challenger that has not cleared the bar is retired
LEAD = 0.25  # it must earn more than 25% more per dollar than the live account over the same days
CONFIDENCE = 0.90  # and the bootstrap must give at least a 90% chance that it is better
BOOT_SAMPLES = 2000  # resamples of its days, drawn with a fixed seed so the verdict is reproducible
BOOT_SEED = 20261105
REVERT_DAYS = 30  # an adopted challenger is checked against the version it replaced over its next 30 days
REVERT_GAP = 0.25  # and is reverted if it trails that version by 25% per dollar
MAX_ACTIVE = 2  # active challengers per account
STATUSES = ("active", "promoted", "retired", "reverted")
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


def verdict(days: int, lead: float, p_better: float) -> str | None:
    """'promote', 'retire' or None for a challenger with `days` trading days of its own, this per-dollar lead and
    this bootstrap probability of being better."""
    if days < MIN_DAYS:
        return None
    if lead > LEAD and p_better >= CONFIDENCE:
        return "promote"
    if lead <= -LEAD or days >= MAX_DAYS:
        return "retire"
    return None


def per_dollar_lead(ret: float, base: float) -> float:
    """How much more `ret` earns per dollar than `base`, as a share of base's result (sign-safe)."""
    return (ret - base) / max(abs(base), 1e-9)


def total(rets) -> float:
    """Return per dollar over a run of daily returns."""
    return float(np.prod([1 + r for r in rets]) - 1) if len(rets) else 0.0


def p_better(pairs) -> float:
    """Share of fixed-seed bootstrap resamples of the days in which the first series out-earns the second."""
    if not pairs:
        return 0.0
    diff = np.array([a - b for a, b in pairs], dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    means = diff[rng.integers(0, len(diff), size=(BOOT_SAMPLES, len(diff)))].mean(axis=1)
    return float((means > 0).mean())


def score(pairs) -> tuple[float, float]:
    """(per-dollar lead, bootstrap probability) of the first series of daily returns against the second."""
    return per_dollar_lead(total([a for a, _ in pairs]), total([b for _, b in pairs])), p_better(pairs)


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
                "p_better": None,
                "rets": [],  # [its daily return, the live account's] for each of its own days
                "prev": None,  # [its equity, the live account's] at the last close
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

    def day(self, runners, d, acct, avg_cost, live_equity, panel, cost_rate, hold=False) -> Runner | None:
        """Step each challenger through day `d`; return the one to promote, if any cleared the bar.

        `hold` (a change is already under watch on the account, or was made today) steps them but promotes none."""
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
                prev = st.get("prev")  # missing only for a challenger started before 2026-11-05
                if prev:
                    st.setdefault("rets", []).append([equity / prev[0] - 1, live_equity / prev[1] - 1])
                st["prev"] = [equity, live_equity]
                st["equity"] = equity
                st["lead"], st["p_better"] = score(st.get("rets", []))
                self.rows.append(
                    {"date": d, "account": st["account"], "challenger": r.id, "equity": equity, "trades": len(fills)}
                )
            else:  # the day it was added or before: no record yet, only a copy of the live book
                r.acct, r.avg_cost = Account(cash=acct.cash, positions=dict(acct.positions)), dict(avg_cost)
                st["prev"] = [live_equity, live_equity]
            st["pending"] = _pending(r.weights, d)
            st["last_date"] = d.date().isoformat()
            st["cash"], st["positions"], st["avg_cost"] = r.acct.cash, dict(r.acct.positions), dict(r.avg_cost)
            if not own:
                continue
            v = verdict(st["days"], st["lead"], st["p_better"])
            if v == "retire":
                if st["lead"] <= -LEAD:
                    why = f"behind the live setting per dollar by {_pct(-st['lead'])} after {st['days']} days"
                else:
                    why = (
                        f"{_pct(st['lead'], sign=True)} against the live setting per dollar, "
                        f"{st['p_better']:.0%} likely better, after {st['days']} days: short of the bar"
                    )
                _end(st, "retired", d, why)
                self.events.append(f"challenger {r.id} retired: {st['result']}")
                runners.remove(r)
            elif v == "promote" and not hold and (best is None or st["lead"] > best.st["lead"]):
                best = r
        return best

    def promote(self, state, runners, r: Runner, a: dict, name: str, d, live_equity: float, acct=None, avg_cost=None):
        """Hand the account to challenger `r` and open the watch on it; return its settings row.

        `acct` and `avg_cost` are the live book at the close of `d`: the replaced version's shadow starts from it,
        with the targets the replaced version holds pending (a["pending"] before the handover).
        """
        st = r.st
        prev = (a.get("following") or {}).get("id")
        if prev in state.get("challengers", {}):
            _end(state["challengers"][prev], "retired", d, f"replaced by {r.id}")
        why = (
            f"ahead of the live setting per dollar by {_pct(st['lead'])} over {st['days']} days, "
            f"{st['p_better']:.0%} likely better"
        )
        _end(st, "promoted", d, why)
        runners.remove(r)
        if acct is not None:
            a["watch"] = {
                "id": r.id,
                "replaced": copy.deepcopy(a.get("following")),  # None: the menu layer's choice
                "replaced_variant": a.get("variant"),
                "cash": acct.cash,
                "positions": dict(acct.positions),
                "avg_cost": dict(avg_cost or {}),
                "pending": copy.deepcopy(a.get("pending")),
                "last_date": pd.Timestamp(d).date().isoformat(),
                "days": 0,
                "rets": [],  # [the adopted version's daily return, the replaced version's]
                "prev": [live_equity, live_equity],
            }
        a["following"] = {"id": r.id, "variant": copy.deepcopy(st["variant"])}
        a["variant"] = f"review:{r.id}"
        self.events.append(f"challenger {r.id} promoted on {name}: {st['result']}")
        reason = (
            f"review challenger {r.id} (rule {st['rule_id']}) ahead of the live setting per dollar by "
            f"{_pct(st['lead'])} over its {st['days']} trading days since {st['started']} (${st['equity']:,.2f} "
            f"against ${live_equity:,.2f}), after costs, {st['p_better']:.0%} likely better by bootstrap; "
            f"hypothesis: {st['hypothesis']}"
        )
        return {"date": d, "account": name, "variant": a["variant"], "description": st["description"], "reason": reason}

    def watch(self, state, a: dict, name: str, kind: str, d, live_equity: float, panel, cost_rate, w_old):
        """Step the version an adopted challenger replaced through day `d`. Returns a settings row on a revert.

        `w_old` is the replaced version's weights. After REVERT_DAYS days the watch closes: the account goes back to
        the replaced version if the adopted one trails it per dollar by REVERT_GAP, and otherwise keeps it.
        """
        wt = a["watch"]
        if d <= pd.Timestamp(wt["last_date"]):
            return None
        shadow = Account(cash=wt["cash"], positions=dict(wt["positions"]))
        cost = dict(wt["avg_cost"])
        simulate.step(shadow, d, panel, _series(wt["pending"]), cost_rate, cost)
        equity = round(simulate.mark(shadow, d, panel), 2)
        wt["rets"].append([live_equity / wt["prev"][0] - 1, equity / wt["prev"][1] - 1])
        wt["prev"] = [live_equity, equity]
        wt["days"] += 1
        wt["pending"] = _pending(w_old, d)
        wt["last_date"] = pd.Timestamp(d).date().isoformat()
        wt["cash"], wt["positions"], wt["avg_cost"] = shadow.cash, dict(shadow.positions), cost
        if wt["days"] < REVERT_DAYS:
            return None
        lead = per_dollar_lead(total([x for x, _ in wt["rets"]]), total([y for _, y in wt["rets"]]))
        st = state.get("challengers", {}).get(wt["id"], {})
        del a["watch"]
        if lead > -REVERT_GAP:
            self.events.append(f"challenger {wt['id']} kept on {name} after {REVERT_DAYS} days ({_pct(lead, True)})")
            return None
        if wt["replaced"]:
            a["following"] = wt["replaced"]
        else:
            a.pop("following", None)
        a["variant"] = wt["replaced_variant"]
        if st:
            _end(
                st, "reverted", d, f"behind the version it replaced per dollar by {_pct(-lead)} over {REVERT_DAYS} days"
            )
            st["status"] = "reverted"
        self.events.append(f"challenger {wt['id']} reverted on {name}")
        reason = (
            f"reverted review challenger {wt['id']}: behind the version it replaced per dollar by {_pct(-lead)} over "
            f"its first {REVERT_DAYS} trading days (${live_equity:,.2f} against ${equity:,.2f}), after costs"
        )
        vid = a["variant"] or ""
        desc = (
            variants.describe_definition(kind, wt["replaced"]["variant"])
            if wt["replaced"]
            else (variants.describe(vid) if vid else "")
        )
        return {"date": d, "account": name, "variant": vid, "description": desc, "reason": reason}

    def close_all(self, runners, d, reason: str) -> None:
        for r in list(runners):
            _end(r.st, "retired", d, reason)
            self.events.append(f"challenger {r.id} retired: {reason}")
            runners.remove(r)


def _pct(x: float, sign: bool = False) -> str:
    if not np.isfinite(x) or abs(x) >= 100:
        return ("+" if sign and x > 0 else "-" if x < 0 and sign else "") + "more than 10000%"
    return f"{x * 100:+.1f}%" if sign else f"{x * 100:.1f}%"


def _end(st: dict, status: str, d, result: str) -> None:
    st["status"], st["ended"], st["result"] = status, str(pd.Timestamp(d).date()), result


def summary(state: dict) -> dict:
    """Every challenger's record so far, and the entries refused, for summary.json."""
    keep = ("account", "description", "rule_id", "started", "status", "days", "equity", "lead", "p_better")
    keep += ("ended", "result")
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
