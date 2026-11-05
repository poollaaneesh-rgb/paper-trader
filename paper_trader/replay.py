"""Loss replay: recompute one night's decision exactly, and try a candidate fix on every other day of the account.

    python -m paper_trader.replay ml_stocks 2026-11-03
    python -m paper_trader.replay ml_stocks 2026-11-03 --fix '{"entry": 0.58, "gap": 0.04, "hold": 5}'

`day` is the close the decision was made at (a loss on day X traces to the decision at the close before it). The
replay reads results/live/decisions.csv for that day's data date, panel digest and code version, rebuilds the frozen
panel through that date (from the working tree, or from the git history of the price files if a dividend rescale has
moved it since), checks the digest, and reruns that code version from `git archive` in a temporary directory, in a
child process with the network switched off. It writes nothing: the live record is read, never changed. If the
recomputed targets differ from the recorded ones, the replay itself is the first defect.

With `--fix` (a setting in the challengers' format, variants.check_definition) it reports whether the fix changes
that day's targets, and the return per dollar of the account over every other settled decision of the account, live
against fixed. A decision's return is its target weights times the move from the next open to the open after (the
days it is held), less costs on the change from the previous targets. The fix passes when it changes the day and
earns more than BAR (25%) more per dollar than the live decisions on those days. Passing earns it a challenger slot,
judged forward only (challengers.py); it never changes the account directly.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from paper_trader import decisions as records
from paper_trader import engines, variants
from paper_trader.config import LIVE_START, MARKETS
from paper_trader.data import Panel, _scaled, load_frozen

BAR = 0.25  # the fix must earn more than 25% more per dollar than the live code on the other days

RUNNER = """
import json, pickle, socket, sys

def _no_network(*args, **kwargs):
    raise OSError("replay sandbox: no network")

socket.socket.connect = _no_network
socket.create_connection = _no_network
sys.path.insert(0, sys.argv[1])
from paper_trader import config, engines
from paper_trader.data import Panel

job = json.loads(open(sys.argv[2]).read())
with open(sys.argv[3], "rb") as fh:
    frames = pickle.load(fh)
panel = Panel(frames["open"], frames["close"], frames["volume"])
market = config.MARKETS[job["market"]]
w, explain, diag = engines.targets(job["kind"], market, panel, config.LIVE_START)
if job["following"]:
    w, explain = engines.challenger_targets(job["kind"], market, panel, job["following"], diag)
import pandas as pd
d = pd.Timestamp(job["day"])
row = w.loc[d] if d in w.index else None
pending = None if row is None or row.isna().all() else row.fillna(0.0).to_dict()
print(json.dumps(pending))
"""


def _git(root: Path, *args: str, binary: bool = False):
    out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode()


def _price_paths(mkey: str) -> tuple[str, str]:
    return f"results/live/prices_{mkey}.csv", f"results/live/prices_{mkey}.scale.json"


def _load_at(root: Path, mkey: str, commit: str | None, tmp: Path) -> Panel | None:
    prices, scale = _price_paths(mkey)
    if commit is None:
        raw, sc = load_frozen(root / prices)
        return None if raw is None else _scaled(raw, sc)
    try:
        (tmp / "prices.csv").write_bytes(_git(root, "show", f"{commit}:{prices}", binary=True))
    except subprocess.CalledProcessError:
        return None
    try:
        (tmp / "prices.scale.json").write_bytes(_git(root, "show", f"{commit}:{scale}", binary=True))
    except subprocess.CalledProcessError:
        (tmp / "prices.scale.json").write_text("{}")
    raw, sc = load_frozen(tmp / "prices.csv")
    return None if raw is None else _scaled(raw, sc)


def rebuild_panel(root: Path, mkey: str, data_date: str, want: str) -> tuple[Panel | None, str]:
    """The panel the night saw, matched by digest: the working tree first, then each commit of the price files."""
    with tempfile.TemporaryDirectory() as tmp:
        candidates: list[str | None] = [None]
        with contextlib.suppress(OSError, subprocess.CalledProcessError):
            candidates += _git(root, "log", "--format=%h", "--", *_price_paths(mkey)).split()
        for commit in candidates:
            full = _load_at(root, mkey, commit, Path(tmp))
            if full is None:
                continue
            panel = full.truncate(data_date)
            if not want or records.digest(panel) == want:
                return panel, "working tree" if commit is None else f"prices at commit {commit}"
    return None, f"no copy of the {mkey} prices through {data_date} matches digest {want}"


def sandbox_targets(root: Path, code: str, kind: str, mkey: str, panel: Panel, day, following) -> dict | None:
    """The targets code version `code` decides at close `day` on `panel`, run apart from the live checkout."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        code_dir = tmp / "code"
        code_dir.mkdir()
        archive = _git(root, "archive", "--format=tar", code, "paper_trader", binary=True)
        subprocess.run(["tar", "-x", "-C", str(code_dir)], input=archive, check=True)
        (tmp / "job.json").write_text(
            json.dumps({"kind": kind, "market": mkey, "day": str(pd.Timestamp(day).date()), "following": following})
        )
        # Pickle keeps every float bit for bit; the file is written here and read only by the child below.
        with open(tmp / "panel.pkl", "wb") as fh:
            pickle.dump({"open": panel.open, "close": panel.close, "volume": panel.volume}, fh)
        (tmp / "runner.py").write_text(RUNNER)
        out = subprocess.run(
            [
                sys.executable,
                "-I",
                str(tmp / "runner.py"),
                str(code_dir),
                str(tmp / "job.json"),
                str(tmp / "panel.pkl"),
            ],
            capture_output=True,
            text=True,
            cwd=tmp,
        )
        if out.returncode != 0:
            raise RuntimeError(f"replay of code {code} failed: {out.stderr.strip()[-800:]}")
        return json.loads(out.stdout.strip().splitlines()[-1])


def _record(root: Path, account: str, day) -> dict | None:
    df = records.load(root / "results" / "live" / "decisions.csv")
    hit = df[(df["account"] == account) & (df["date"] == str(pd.Timestamp(day).date()))]
    return None if hit.empty else hit.iloc[-1].to_dict()


def reproduce(root: Path, account: str, day) -> dict:
    """Recompute the decision at close `day` from its record. Returns what was found; changes nothing."""
    mkey, kind = engines.ACCOUNTS[account]
    rec = _record(root, account, day)
    if rec is None:
        return {"reproduced": None, "note": f"no decision record for {account} on {pd.Timestamp(day).date()}"}
    panel, source = rebuild_panel(root, mkey, rec["data_date"], rec["data_digest"])
    if panel is None:
        return {"reproduced": False, "note": source, "record": rec}
    following = json.loads(rec["following"]) if rec["following"] else None
    got = sandbox_targets(root, rec["code"], kind, mkey, panel, day, following)
    want = records.parse_targets(rec["targets"])
    ok = records.same_targets(got, want)
    return {"reproduced": ok, "note": f"data from the {source}, code {rec['code']}", "record": rec, "targets": got}


def _decision_returns(panel: Panel, targets: dict, cost_rate: float) -> pd.Series:
    """Per settled decision date: target weights times next-open to following-open move, less cost of the change."""
    opens = panel.open.ffill()
    dates = list(panel.dates)
    out, prev = {}, {}
    for d in sorted(targets):
        p = targets[d]
        p = prev if p is None else p
        i = dates.index(d)
        if i + 2 >= len(dates):
            continue
        move = opens.iloc[i + 2] / opens.iloc[i + 1] - 1
        gross = sum(w * float(move.get(t, 0.0)) for t, w in p.items() if w)
        turnover = sum(abs(p.get(t, 0.0) - prev.get(t, 0.0)) for t in set(p) | set(prev))
        out[d] = gross - cost_rate * turnover
        prev = p
    return pd.Series(out, dtype=float)


def per_dollar_lead(fixed: float, live: float) -> float:
    """How much more the fix earns per dollar than live, as a share of live's result (sign-safe)."""
    return (fixed - live) / max(abs(live), 1e-9)


def try_fix(root: Path, account: str, day, fix: dict, panel: Panel | None = None) -> dict:
    """Score `fix` against the live decisions on every settled decision of `account` except the one at `day`."""
    mkey, kind = engines.ACCOUNTS[account]
    bad = variants.check_definition(kind, fix)
    if bad:
        raise ValueError(bad)
    market = MARKETS[mkey]
    if panel is None:
        raw, sc = load_frozen(root / "results" / "live" / f"prices_{mkey}.csv")
        panel = _scaled(raw, sc)
    day = pd.Timestamp(day)
    w_menu, _, diag = engines.targets(kind, market, panel, LIVE_START)
    w_fix, _ = engines.challenger_targets(kind, market, panel, fix, diag)
    df = records.load(root / "results" / "live" / "decisions.csv")
    df = df[df["account"] == account]
    recorded = {pd.Timestamp(r["date"]): records.parse_targets(r["targets"]) for r in df.to_dict("records")}
    start = panel.dates[panel.dates < pd.Timestamp(LIVE_START)][-1]
    days = [d for d in panel.dates if d >= start]

    def row(w, d):
        r = w.loc[d] if d in w.index else None
        return None if r is None or r.isna().all() else r.fillna(0.0).to_dict()

    live = {d: recorded[d] if d in recorded else row(w_menu, d) for d in days}
    fixed = {d: row(w_fix, d) for d in days}
    r_live = _decision_returns(panel, live, market.cost_rate)
    r_fix = _decision_returns(panel, fixed, market.cost_rate)
    others = [d for d in r_live.index if d != day and d in r_fix.index]
    live_total, fix_total = float(r_live[others].sum()), float(r_fix[others].sum())
    changes = not records.same_targets(live.get(day), fixed.get(day))
    lead = per_dollar_lead(fix_total, live_total)
    return {
        "changes_day": changes,
        "days": len(others),
        "unrecorded_days": sum(1 for d in days if d not in recorded),
        "live": live_total,
        "fixed": fix_total,
        "lead": lead,
        "passes": bool(changes and lead > BAR),
    }


def main(argv=None, root: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m paper_trader.replay", description=__doc__.split("\n\n")[0])
    ap.add_argument("account", choices=sorted(engines.ACCOUNTS))
    ap.add_argument("day", help="the close the decision was made at, YYYY-MM-DD")
    ap.add_argument("--fix", help="a candidate setting as JSON, in the challengers' format")
    args = ap.parse_args(argv)
    root = root or Path(__file__).resolve().parents[1]
    res = reproduce(root, args.account, args.day)
    if res["reproduced"] is None:
        print(f"not reproducible: {res['note']}")
    elif res["reproduced"]:
        print(f"reproduced: {args.account} at the close of {args.day} ({res['note']})")
    else:
        print(f"DOES NOT REPRODUCE, the replay is the first defect: {res['note']}")
        if "targets" in res:
            print(f"  recorded: {res['record']['targets'] or 'hold'}")
            print(f"  replayed: {records.targets_text(res['targets']) or 'hold'}")
    if args.fix:
        out = try_fix(root, args.account, args.day, json.loads(args.fix))
        print(f"fix changes {args.day}: {'yes' if out['changes_day'] else 'no'}")
        print(
            f"return per dollar over the other {out['days']} settled decisions: live {out['live'] * 100:+.2f}%, "
            f"fixed {out['fixed'] * 100:+.2f}% ({out['lead'] * 100:+.0f}% against live; the bar is more than "
            f"{BAR:.0%})"
        )
        if out["unrecorded_days"]:
            print(f"  {out['unrecorded_days']} days predate the decision records; their live targets are recomputed")
        print("passes: may enter as a challenger" if out["passes"] else "does not pass: a dated note, not a fix")
    return 0 if res["reproduced"] in (True, None) else 1


if __name__ == "__main__":
    sys.exit(main())
