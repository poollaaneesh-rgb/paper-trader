# Paper Trader

[![CI](https://github.com/poollaaneesh-rgb/paper-trader/actions/workflows/ci.yml/badge.svg)](https://github.com/poollaaneesh-rgb/paper-trader/actions/workflows/ci.yml)

A strategy tournament and a machine-learning model each trade **$400 of fake money**, once a day, in US stocks and in crypto, and have to beat buy-and-hold. Results: **https://poollaaneesh-rgb.github.io/paper-trader/**

Fake money only. Not connected to any brokerage, holds no keys, and is not investment advice.

## Accounts

| Account | Market | Engine |
|---|---|---|
| `tournament_stocks`, `tournament_crypto` | 30 US stocks/ETFs, 6 coins | Momentum, mean reversion and trend; capital shifts monthly toward the winners (10% floor each) |
| `ml_stocks`, `ml_crypto` | same | Gradient-boosted trees on nine price features, retrained weekly; holds the top picks above 55% odds |
| `bench_spy`, `bench_btc` | SPY, BTC | Buy and hold, never trades |

## Method

- **No look-ahead:** decisions use data to the close and fill at the next open. Tests truncate the data and check that no earlier decision changes.
- **Costs:** 0.05% slippage on stocks; 0.5% fee plus 0.05% slippage on crypto.
- **Backtest:** walk-forward 2018-01-01 to 2026-09-30 on prices frozen in `results/backtest/` (Yahoo's adjusted prices shift by about 1e-6 between downloads, which was enough to move the stock ML result between $484 and $965).
- **Live paper:** from 2026-10-01, one day at a time, by GitHub Actions every evening. Only this record is real evidence, and the live record is never recomputed. One known deviation: the crypto accounts were set up before Sep 30 had closed, so their first fill landed on Sep 30, a day early. Fixed on 2026-10-04; the recorded days stay as they happened.

## Findings so far (backtest)

- The tournament nearly matched the S&P 500 on stocks ($1,255 vs $1,300) with a smaller worst drop (-25% vs -34%) and a higher Sharpe ratio (0.91 vs 0.81). None of that is evidence of skill: the daily return difference from SPY has a t-statistic of -0.27, each Sharpe ratio carries a standard error of about 0.40 over this sample, and the smaller drop mostly reflects a beta of 0.69 to the market. It is one sample period, and the strategy thresholds were never tested out of sample.
- The ML model's daily up/down calls on stocks are right 53.2% of the time, slightly worse than always guessing "up" (stocks rose on 53.5% of the scored days). It turns over 57% of the stock account a day. Fees decide its result: $2,083 with no fees, $597 at 0.05%, $171 at 0.1%. In crypto, 0.55% per trade took it from $3,620 (no fees) to $2.75.
- Survivorship bias: the universe is today's large names, which flatters every backtest here.

## Code map

| File | What it does |
|---|---|
| `paper_trader/strategies.py` | Momentum, mean reversion and trend, each a function from prices to target weights |
| `paper_trader/tournament.py` | Monthly capital allocation across the three strategies by trailing Sharpe |
| `paper_trader/features.py`, `ml.py` | Nine price features, the forward label, weekly walk-forward retraining |
| `paper_trader/portfolio.py`, `simulate.py` | Cash-and-units account with costs; next-open fills |
| `paper_trader/live.py`, `data.py` | Nightly step-forward with stale-data guards; yfinance plus Coinbase fallback |
| `paper_trader/metrics.py`, `report.py` | Scorecard numbers and the static results page |
| `bquant/` | The stock tournament rerun on Bloomberg prices, for comparison |

Design decisions and known limits: [docs/design.md](docs/design.md).

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/pip install ruff && .venv/bin/ruff check .
.venv/bin/python scripts/backtest.py     # about 2.5 minutes; uses the frozen prices
.venv/bin/python scripts/run_daily.py    # steps the live accounts and rebuilds site/
```
