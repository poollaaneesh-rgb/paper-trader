"""Nightly live paper-trading run (GitHub Actions, about 9 PM Arizona). Fake money only."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_trader import data, engines, live, metrics, report
from paper_trader.config import DATA_START, LIVE_START, MARKETS, START_CASH

RES = ROOT / "results"
LIVE = RES / "live"


def _append(path: Path, rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def settings_summary() -> dict:
    """Each account's current setting and the log of changes, from results/live/settings.csv."""
    path = LIVE / "settings.csv"
    if not path.exists():
        return {"current": {}, "changes": []}
    rows = pd.read_csv(path).sort_values("date")
    current = {
        r["account"]: {"variant": r["variant"], "description": r["description"], "since": r["date"]}
        for r in rows.to_dict("records")
    }
    return {"current": current, "changes": rows.iloc[::-1].head(20).to_dict("records")}


def live_summary(skips, states: dict | None = None) -> dict:
    eq = pd.read_csv(LIVE / "equity.csv", parse_dates=["date"]) if (LIVE / "equity.csv").exists() else None
    tr = pd.read_csv(LIVE / "trades.csv") if (LIVE / "trades.csv").exists() else pd.DataFrame()
    out = {"phase": "live paper", "start": LIVE_START, "start_cash": START_CASH, "accounts": {}, "skips": skips}
    out["settings"] = settings_summary()
    for name, (mkey, kind) in engines.ACCOUNTS.items():
        series = eq[eq.account == name].set_index("date")["equity"] if eq is not None else pd.Series(dtype=float)
        trades = tr[tr.account == name] if len(tr) else pd.DataFrame(columns=["realized_pnl"])
        s = metrics.summary(series, trades, 252 if mkey == "stocks" else 365, START_CASH)
        s.update({"market": mkey, "engine": kind, "bankrupt": (states or {}).get(name, {}).get("bankrupt")})
        out["accounts"][name] = s
    return out


def main():
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    LIVE.mkdir(parents=True, exist_ok=True)
    state_path = LIVE / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else live.init_state()
    panels = {}
    for k, m in MARKETS.items():
        try:
            panels[k] = data.complete_days(data.fetch(m, DATA_START), today)
        except Exception as exc:  # a failed source skips that market; it never trades on bad data
            print(f"{k}: fetch failed: {exc}")
            panels[k] = None
    state, eq_rows, trade_rows, setting_rows = live.run(state, panels, today)
    _append(LIVE / "equity.csv", eq_rows)
    _append(LIVE / "trades.csv", trade_rows)
    _append(LIVE / "settings.csv", setting_rows)
    state_path.write_text(json.dumps(state, indent=2, default=str))
    backtest = json.loads((RES / "backtest" / "summary.json").read_text())
    summary = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes"),
        "live": live_summary(state["skips"], state["accounts"]),
        "backtest": backtest,
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    report.build(RES, ROOT / "site")
    print(f"processed {len(eq_rows)} account-days, {len(trade_rows)} trades, {len(setting_rows)} setting changes")


if __name__ == "__main__":
    main()
