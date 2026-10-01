"""Builds the public results page (site/index.html) and the dashboard feed (site/summary.json)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LABELS = {
    "tournament_stocks": "Tournament", "ml_stocks": "ML model", "bench_spy": "S&P 500 buy-and-hold",
    "tournament_crypto": "Tournament", "ml_crypto": "ML model", "bench_btc": "Bitcoin buy-and-hold",
}
MARKET_ACCOUNTS = {"stocks": ["tournament_stocks", "ml_stocks", "bench_spy"],
                   "crypto": ["tournament_crypto", "ml_crypto", "bench_btc"]}


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
    payload = {
        "summary": summary,
        "updated": _when(summary["generated_at"]),
        "labels": LABELS,
        "markets": MARKET_ACCOUNTS,
        "backtest_curves": _curves(results / "backtest" / "equity.csv", weekly=True),
        "live_curves": _curves(results / "live" / "equity.csv", weekly=False),
        "trades": live_trades or _trades(results / "backtest" / "trades.csv", 30, "backtest"),
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(payload, default=str).replace("</", "<\\/"))
    (site / "index.html").write_text(html)
    feed = {
        "generated_at": summary["generated_at"],
        "page": "https://poollaaneesh-rgb.github.io/paper-trader/",
        "live_start": summary["live"]["start"],
        "live": {k: {f: v.get(f) for f in ("final_equity", "total_return", "days", "n_trades")}
                 for k, v in summary["live"]["accounts"].items()},
        "backtest": {k: {f: v.get(f) for f in ("final_equity", "total_return")}
                     for k, v in summary["backtest"]["accounts"].items()},
        "latest_trades": live_trades[:5],
        "skips": summary["live"].get("skips", [])[-3:],
    }
    (site / "summary.json").write_text(json.dumps(feed, indent=1, default=str))


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper Trader</title>
<meta name="description" content="Two self-learning strategies trade $400 of fake money each, every day, against buy-and-hold.">
<style>
:root{--bg:#ffffff;--ink:#15181e;--ink2:#4a5060;--muted:#6b7280;--line:#e3e5ea;--navy:#0b2545;--red:#c8102e;
--s1:#1c5cab;--s2:#eb6834;--ref:#6b7280}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Aptos",-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-variant-numeric:tabular-nums}
main{max-width:1040px;margin:0 auto;padding:40px 16px 64px}
h1{font-size:40px;line-height:1.1;margin:0 0 8px;color:var(--navy);letter-spacing:-.01em}
h2{font-size:24px;margin:48px 0 4px;color:var(--navy)}
h3{font-size:17px;margin:28px 0 8px}
.lede{font-size:17px;color:var(--ink2);max-width:720px;margin:0}
.meta{font-size:13px;color:var(--muted);margin-top:8px}
.label{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.point{border-left:3px solid var(--red);padding:2px 0 2px 12px;margin:16px 0;font-size:17px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:12px}
.stat{padding:12px 0;border-top:1px solid var(--line)}
.stat .v{font-size:32px;line-height:1.1;font-weight:600}
.stat .s{font-size:13px;color:var(--ink2)}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{width:100%;border-collapse:collapse;font-size:14px;margin-top:8px}
th{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);text-align:right;font-weight:600;padding:8px 10px;border-bottom:1px solid var(--navy)}
td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td.reason{text-align:left;white-space:normal;color:var(--ink2);min-width:260px}
.key{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:8px}
.key.dash{background:repeating-linear-gradient(90deg,var(--ref) 0 4px,transparent 4px 7px)}
.chart{position:relative;height:320px;margin-top:8px}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--ink2);margin-top:4px}
.empty{color:var(--muted);font-size:14px;padding:12px 0;border-top:1px solid var(--line)}
ul{padding-left:20px;max-width:760px}li{margin:6px 0}
footer{margin-top:56px;font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:12px}
@media (max-width:600px){h1{font-size:32px}.stat .v{font-size:24px}.chart{height:260px}}
</style></head><body><main>
<h1>Paper Trader</h1>
<p class="lede">A strategy tournament and a machine-learning model each trade $400 of fake money, once a day, in US stocks and in crypto. Each has to beat simply buying and holding the index. Built by Aneesh Poolla.</p>
<p class="meta" id="updated"></p>

<h2>Live paper trading</h2>
<p class="label" id="live-since"></p>
<div id="live"></div>

<h2>Backtest</h2>
<p class="label">Walk-forward, Jan 2018 to Sep 2026 · fees and slippage included · weekly points</p>
<div id="backtest"></div>

<h2>Latest decisions</h2>
<p class="label" id="trades-label"></p>
<div class="scroll"><table id="trades"><thead><tr><th>Date</th><th>Account</th><th>Action</th><th>Value</th><th>Why</th></tr></thead><tbody></tbody></table></div>

<h2>How it works</h2>
<ul>
<li><b>Tournament.</b> Momentum (top 3-month returns), mean reversion (5-day RSI under 30, sell above 50) and trend (above the 50-day average, 50-day above the 200-day) each run a share of the account. On the first trading day of every month, money shifts toward the strategies with the best risk-adjusted return over the last three months, with a 10% floor each.</li>
<li><b>ML model.</b> Gradient-boosted trees on nine price features (returns over 1 to 60 days, volatility, RSI, distance from moving averages, volume change) predict whether each asset rises over the next holding day. Retrained every week on data up to two days before, then holds the top picks above 55% odds. A logistic regression runs alongside as a sanity check.</li>
<li><b>No look-ahead.</b> Decisions use data up to the close; orders fill at the next day's open. Automated tests truncate the data and check that no past decision changes.</li>
<li><b>Costs.</b> Stocks pay 0.05% slippage per trade; crypto pays 0.5% fees plus slippage.</li>
<li><b>Known limits.</b> The universe is today's large companies and coins, which flatters any backtest (survivorship bias). Only the live paper record is real evidence.</li>
</ul>
<footer>Fake money only. This is a research project, not investment advice, and it is not connected to any brokerage. Code: <a href="https://github.com/poollaaneesh-rgb/paper-trader">github.com/poollaaneesh-rgb/paper-trader</a></footer>
</main>
<script id="data" type="application/json">__DATA__</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COLORS = {tournament: css('--s1'), ml: css('--s2'), bench: css('--ref')};
const money = v => v == null ? '–' : '$' + Number(v).toLocaleString('en-US', {maximumFractionDigits: 0});
const pct = v => v == null ? '–' : (v >= 0 ? '+' : '−') + Math.abs(v * 100).toFixed(1) + '%';
const num = (v, d = 2) => v == null ? '–' : Number(v).toFixed(d);
const kindOf = n => n.startsWith('ml') ? 'ml' : n.startsWith('bench') ? 'bench' : 'tournament';
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.getElementById('updated').textContent = 'Updated ' + D.updated;
const live = D.summary.live, bt = D.summary.backtest;
const liveStart = new Date(live.start + 'T12:00:00').toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric'});
document.getElementById('live-since').textContent = 'Since ' + liveStart + ' · starts at $400 each · one real day at a time';

function verdict(accts, market) {
  const [t, m, b] = D.markets[market].map(n => accts[n]);
  if (!b || !b.days) return null;
  const beat = [['Tournament', t], ['ML model', m]].filter(([, a]) => a.total_return > b.total_return).map(([n]) => n);
  const bench = D.labels[D.markets[market][2]];
  if (beat.length === 2) return 'Both bots beat ' + bench + ' (' + pct(b.total_return) + ').';
  if (beat.length === 1) return beat[0] + ' beat ' + bench + ' (' + pct(b.total_return) + '); the other did not.';
  return 'Neither bot beat ' + bench + ' (' + pct(b.total_return) + ').';
}

function chart(el, curves, names) {
  const series = names.filter(n => curves[n]);
  if (!series.length) return;
  const labels = curves[series[series.length - 1]].dates;
  const endLabels = {id: 'endLabels', afterDatasetsDraw(c) {
    const ctx = c.ctx; ctx.save(); ctx.font = '12px -apple-system, Segoe UI, sans-serif'; ctx.fillStyle = css('--ink2');
    c.data.datasets.forEach((ds, i) => { const meta = c.getDatasetMeta(i); const p = meta.data[meta.data.length - 1];
      if (p) ctx.fillText(ds.short, Math.min(p.x + 6, c.width - 70), p.y + 4); });
    ctx.restore(); }};
  new Chart(el, {type: 'line', plugins: [endLabels],
    data: {labels, datasets: series.map(n => {
      const map = Object.fromEntries(curves[n].dates.map((d, i) => [d, curves[n].equity[i]]));
      const k = kindOf(n);
      return {label: D.labels[n], short: k === 'bench' ? 'Hold' : k === 'ml' ? 'ML' : 'Tournament',
        data: labels.map(d => map[d] ?? null), borderColor: COLORS[k], borderWidth: 2, pointRadius: 0,
        pointHoverRadius: 4, borderDash: k === 'bench' ? [4, 3] : [], spanGaps: true, tension: 0};
    })},
    options: {responsive: true, maintainAspectRatio: false, animation: false,
      layout: {padding: {right: 72}},
      interaction: {mode: 'index', intersect: false},
      plugins: {legend: {display: false}, tooltip: {callbacks: {label: c => c.dataset.label + ': ' + money(c.parsed.y)}}},
      scales: {x: {ticks: {maxTicksLimit: 6, color: css('--muted'), callback(v) { const d = this.getLabelForValue(v);
                 return new Date(d + 'T12:00:00').toLocaleDateString('en-US', {month: 'short', year: 'numeric'}); }},
                   grid: {display: false}},
               y: {ticks: {color: css('--muted'), callback: v => money(v)}, grid: {color: css('--line')}}}}});
}

function legend(names) {
  return '<div class="legend">' + names.map(n => { const k = kindOf(n);
    return '<span><span class="key' + (k === 'bench' ? ' dash' : '') + '" style="' + (k === 'bench' ? '' : 'background:' + COLORS[k]) + '"></span>' + esc(D.labels[n]) + '</span>'; }).join('') + '</div>';
}

function section(rootId, block, curves, phase) {
  const root = document.getElementById(rootId);
  for (const market of ['stocks', 'crypto']) {
    const names = D.markets[market];
    const accts = block.accounts;
    const title = market === 'stocks' ? 'US stocks and ETFs' : 'Crypto';
    let html = '<h3>' + title + '</h3>';
    const started = names.some(n => accts[n] && accts[n].days > 0);
    if (!started) {
      html += '<p class="empty">No complete trading days yet. The first fills happen at the open on ' + liveStart + '; results appear after that day closes.</p>';
      root.insertAdjacentHTML('beforeend', html); continue;
    }
    const v = verdict(accts, market);
    if (v) html += '<p class="point">' + esc(v) + '</p>';
    html += '<div class="grid">' + names.map(n => '<div class="stat"><div class="label">' + esc(D.labels[n]) + '</div><div class="v">' + money(accts[n].final_equity) + '</div><div class="s">' + pct(accts[n].total_return) + ' from $400</div></div>').join('') + '</div>';
    const id = rootId + '-' + market;
    html += '<div class="chart"><canvas id="' + id + '" role="img" aria-label="' + esc(title) + ' account balance over time"></canvas></div>' + legend(names);
    html += '<div class="scroll"><table><thead><tr><th>Account</th><th>Balance</th><th>Return</th><th>Worst drop</th><th>Sharpe</th><th>Win rate</th><th>Trades</th></tr></thead><tbody>' +
      names.map(n => { const a = accts[n]; return '<tr><td>' + esc(D.labels[n]) + '</td><td>' + money(a.final_equity) + '</td><td>' + pct(a.total_return) + '</td><td>' + pct(a.max_drawdown) + '</td><td>' + num(a.sharpe) + '</td><td>' + (a.win_rate == null ? '–' : Math.round(a.win_rate * 100) + '%') + '</td><td>' + a.n_trades + '</td></tr>'; }).join('') + '</tbody></table></div>';
    const ml = accts[names[1]].diagnostics;
    if (ml && ml.hgb_hit_rate != null) html += '<p class="meta">ML direction hit rate ' + Math.round(ml.hgb_hit_rate * 1000) / 10 + '% (logistic baseline ' + Math.round(ml.lr_hit_rate * 1000) / 10 + '%, a coin flip is 50%) over ' + ml.predictions_scored.toLocaleString() + ' predictions, ' + ml.retrains + ' weekly retrains.</p>';
    root.insertAdjacentHTML('beforeend', html);
    chart(document.getElementById(id), curves, names);
  }
}
section('live', live, D.live_curves, 'live');
section('backtest', bt, D.backtest_curves, 'backtest');
if (live.skips && live.skips.length) {
  const s = live.skips[live.skips.length - 1];
  document.getElementById('live').insertAdjacentHTML('beforeend', '<p class="meta">Last skipped run: ' + esc(s.run) + ', ' + esc(s.market) + ' (' + esc(s.reason) + ').</p>');
}
const phase = D.trades.length ? D.trades[0].phase : 'live paper';
document.getElementById('trades-label').textContent = phase === 'backtest' ? 'Last 30 backtest trades (live trades replace these once they start)' : 'Last 30 live paper trades';
document.querySelector('#trades tbody').innerHTML = D.trades.map(t => '<tr><td>' + new Date(t.date + 'T12:00:00').toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric'}) + '</td><td>' + esc(D.labels[t.account]) + (t.account.endsWith('crypto') || t.account === 'bench_btc' ? ' · crypto' : ' · stocks') + '</td><td>' + (t.side === 'buy' ? 'Buy ' : 'Sell ') + esc(t.ticker) + '</td><td>' + money(t.value) + '</td><td class="reason">' + esc(t.reason) + '</td></tr>').join('') || '<tr><td colspan="5" class="empty">No trades yet.</td></tr>';
</script></body></html>
"""
