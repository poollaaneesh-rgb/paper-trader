"""Builds the public results page (site/index.html) and the dashboard feed (site/summary.json)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

TEMPLATE = (Path(__file__).parent / "page_template.html").read_text()

LABELS = {
    "tournament_stocks": "Tournament",
    "ml_stocks": "ML model",
    "bench_spy": "S&P 500 buy-and-hold",
    "tournament_crypto": "Tournament",
    "ml_crypto": "ML model",
    "bench_btc": "Bitcoin buy-and-hold",
}
MARKET_ACCOUNTS = {
    "stocks": ["tournament_stocks", "ml_stocks", "bench_spy"],
    "crypto": ["tournament_crypto", "ml_crypto", "bench_btc"],
}


def _curves(path: Path, weekly: bool) -> dict:
    if not path.exists():
        return {}
    eq = pd.read_csv(path, parse_dates=["date"])
    out = {}
    for name, g in eq.groupby("account"):
        s = g.set_index("date")["equity"]
        if weekly:
            s = s.resample("W-FRI").last().dropna()
        out[name] = {"dates": [d.strftime("%Y-%m-%d") for d in s.index], "equity": s.round(2).tolist()}
    return out


def _trades(path: Path, n: int, phase: str) -> list[dict]:
    if not path.exists():
        return []
    tr = pd.read_csv(path)
    if tr.empty:
        return []
    tr = tr.sort_values("date").tail(n).iloc[::-1]
    cols = ["date", "account", "side", "ticker", "value", "reason"]
    return [dict(r, phase=phase, value=round(float(r["value"]), 2)) for r in tr[cols].to_dict("records")]


def _when(iso: str) -> str:
    t = pd.Timestamp(iso).tz_convert("America/Phoenix")
    return t.strftime("%b %-d, %Y, %-I:%M %p") + " Arizona time"


def build(results: Path, site: Path):
    site.mkdir(parents=True, exist_ok=True)
    summary = json.loads((results / "summary.json").read_text())
    live_trades = _trades(results / "live" / "trades.csv", 30, "live paper")
    controls_path = results / "backtest" / "controls.json"
    payload = {
        "summary": summary,
        "updated": _when(summary["generated_at"]),
        "labels": LABELS,
        "markets": MARKET_ACCOUNTS,
        "backtest_curves": _curves(results / "backtest" / "equity.csv", weekly=True),
        "live_curves": _curves(results / "live" / "equity.csv", weekly=False),
        "trades": live_trades or _trades(results / "backtest" / "trades.csv", 30, "backtest"),
        "controls": json.loads(controls_path.read_text()) if controls_path.exists() else {},
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(payload, default=str).replace("</", "<\\/"))
    (site / "index.html").write_text(html)
    feed = {
        "generated_at": summary["generated_at"],
        "page": "https://poollaaneesh-rgb.github.io/paper-trader/",
        "live_start": summary["live"]["start"],
        "live": {
            k: {f: v.get(f) for f in ("final_equity", "total_return", "days", "n_trades")}
            for k, v in summary["live"]["accounts"].items()
        },
        "backtest": {
            k: {f: v.get(f) for f in ("final_equity", "total_return")}
            for k, v in summary["backtest"]["accounts"].items()
        },
        "latest_trades": live_trades[:5],
        "skips": summary["live"].get("skips", [])[-3:],
        "settings": summary["live"].get("settings", {}),
    }
    (site / "summary.json").write_text(json.dumps(feed, indent=1, default=str))
