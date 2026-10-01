# Paper Trader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fake-money trading bot (strategy tournament and ML, stocks and crypto) that backtests from 2018, paper-trades daily from 2026-10-01 on GitHub Actions, and publishes an honest scorecard.

**Architecture:** Every engine is a pure function from price history to a table of target weights (dates x tickers), so look-ahead can be tested by truncating data. A small simulator steps an account through the days: weights decided at close *t* fill at open *t+1*, with costs. The backtest is run once and frozen; the live run keeps account state in `results/live/state.json` and steps forward one day at a time.

**Tech Stack:** Python 3.11, pandas 3, numpy, scikit-learn, yfinance, requests, pytest; static HTML + Chart.js (cdnjs) for the page; GitHub Actions + Pages.

**Spec:** `docs/superpowers/specs/2026-09-30-paper-trader-design.md`

**Execution note (2026-09-30):** executed inline in the same session that wrote it. To save the user's Claude usage, implementation code is written straight into the files rather than duplicated here. Each task below fixes the files, interfaces and tests (the tests are the contract).

## Global Constraints

- Fake money only. No brokerage connection, no order API, no secrets in the repo.
- Six accounts, $400 each: `tournament_stocks`, `tournament_crypto`, `ml_stocks`, `ml_crypto`, `bench_spy`, `bench_btc`.
- Backtest 2018-01-01 to 2026-09-30; live paper from 2026-10-01; always labelled.
- Decide at close *t*, fill at open *t+1*. Costs: stocks 0.05% slippage; crypto 0.5% fee + 0.05% slippage.
- ML retrains weekly (`RETRAIN_EVERY = "W"`). Tournament reweights monthly with a 10% floor.
- Only days strictly before today's UTC date are processed (no partial candles).

---

### Task 1: Config, data and portfolio

**Files:** `paper_trader/config.py`, `paper_trader/data.py`, `paper_trader/portfolio.py`, `tests/test_portfolio.py`, `tests/test_data.py`, `requirements.txt`, `pyproject.toml`

**Interfaces (produces):**
- `config.MARKETS: dict[str, Market]` with `Market(name, tickers, benchmark, cost_rate, max_positions)`
- `data.Panel(open: DataFrame, close: DataFrame, volume: DataFrame)`; `data.fetch(market, start) -> Panel`; `data.complete_days(panel, today_utc) -> Panel`
- `portfolio.Account(cash, positions: dict[str, float])`; `Account.equity(prices: Series) -> float`; `Account.rebalance(weights: Series, prices: Series, cost_rate: float) -> list[Trade]`; `Trade(ticker, units, price, cost)`

**Tests:**
- round trip at a fixed price loses exactly `2 * cost_rate * value` (within 1e-9)
- cash never negative after any rebalance, including weights summing to 1
- trades under max($1, 1% of equity) are skipped
- a NaN price for a ticker leaves that position untouched
- `complete_days` drops rows dated today or later

### Task 2: Strategies and tournament

**Files:** `paper_trader/strategies.py`, `paper_trader/tournament.py`, `tests/test_strategies.py`

**Interfaces:**
- `momentum(panel, max_positions) -> DataFrame`, `mean_reversion(panel, max_positions) -> DataFrame`, `trend(panel, max_positions) -> DataFrame` (rows sum to <= 1, non-negative)
- `tournament.weights(panel, max_positions) -> (weights DataFrame, allocations DataFrame)`
- `strategy_returns(weights, panel) -> Series` = `(weights.shift(2) * open_to_open_returns).sum(axis=1)`

**Tests:**
- **no look-ahead:** for each function, the weight row at date *t* computed on the full panel equals the row computed on the panel truncated at *t*
- rows sum to at most 1 and contain no negatives
- allocations sum to 1, each >= 0.10, and change only on the first trading day of a month

### Task 3: ML engine

**Files:** `paper_trader/features.py`, `paper_trader/ml.py`, `tests/test_ml.py`

**Interfaces:**
- `features.build(panel) -> DataFrame` (MultiIndex date, ticker; columns ret1, ret5, ret20, ret60, vol20, rsi14, dist50, dist200, volchg)
- `features.labels(panel) -> Series` (open[t+2] / open[t+1] - 1 > 0)
- `ml.weights(panel, max_positions, start, retrain="W") -> (weights DataFrame, diagnostics dict)`

**Tests:**
- no look-ahead on a synthetic panel (truncation test, as in Task 2)
- training rows for a retrain on boundary *b* all have date index <= index(b) - 2
- holds nothing when every probability is <= 0.55

### Task 4: Simulator, metrics and backtest

**Files:** `paper_trader/simulate.py`, `paper_trader/metrics.py`, `paper_trader/engines.py`, `scripts/backtest.py`, `tests/test_simulate.py`

**Interfaces:**
- `simulate.step(account, day, panel, target: Series | None, cost_rate) -> list[Trade]` (fills pending target at the day's open)
- `simulate.run(panel, weights, start, end, cash, cost_rate) -> (equity Series, trades DataFrame, Account)`
- `metrics.summary(equity, trades) -> dict` (total_return, max_drawdown, sharpe, win_rate, n_trades)
- `engines.ACCOUNTS: dict[str, (market, weight_fn)]`

**Tests:**
- benchmark account trades exactly once
- equity on each day = cash + units x close
- a perfect-foresight weight table run through `run` with weights shifted correctly cannot earn on day one (fill happens next open)

Backtest run writes `results/backtest/{equity.csv,trades.csv,summary.json}`.

### Task 5: Live daily run, report page, Actions

**Files:** `scripts/run_daily.py`, `paper_trader/report.py`, `site/index.html` (generated), `.github/workflows/daily.yml`, `README.md`, `tests/test_live.py`

**Behaviour:** load `results/live/state.json` (or initialise $400 accounts deciding at the 2026-09-30 close); for each market whose data is complete through yesterday UTC, process each new day: fill pending targets at the open, mark to close, decide the next targets. If data is stale, write a skip reason. Write `results/live/*`, `results/summary.json` (consumed by the dashboard card), rebuild `site/`.

**Tests:**
- running the daily step twice on the same data changes nothing (idempotent)
- a stale market is skipped with a reason and its state is unchanged

### Task 6: Dashboard card

**Files (life-dashboard repo, branch `hummingbird-build`):** a Hummingbird card reading the public `summary.json`. Load `~/Code/life-dashboard/design/hummingbird/QUICK-REFERENCE.md` first.
