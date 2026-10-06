# Paper Trader

[![CI](https://github.com/poollaaneesh-rgb/paper-trader/actions/workflows/ci.yml/badge.svg)](https://github.com/poollaaneesh-rgb/paper-trader/actions/workflows/ci.yml)

A strategy tournament and a machine-learning model each trade **$400 of fake money**, once a day, in US stocks and in crypto, and have to beat buy-and-hold. Since October 2026 each one also tunes its own settings on evidence (below). Results: **https://poollaaneesh-rgb.github.io/paper-trader/**

Fake money only. Not connected to any brokerage, holds no keys, and is not investment advice.

## Accounts

| Account | Market | Engine |
|---|---|---|
| `tournament_stocks`, `tournament_crypto` | 30 US stocks/ETFs, 6 coins | Momentum, mean reversion and trend; capital shifts toward the winners; the self-tuning layer picks the strategy speeds, lookback and cadence |
| `ml_stocks`, `ml_crypto` | same | Gradient-boosted trees on nine price features, retrained weekly; the self-tuning layer picks how the odds become positions (entry line, exit line, minimum hold, or cash) |
| `bench_spy`, `bench_btc` | SPY, BTC | Buy and hold, never trades |

## Method

- **No look-ahead:** decisions use data to the close and fill at the next open. Tests truncate the data and check that no earlier decision changes.
- **Costs:** 0.05% slippage on stocks; 0.5% fee plus 0.05% slippage on crypto.
- **Backtest:** walk-forward 2018-01-01 to 2026-09-30 on prices frozen in `results/backtest/` (Yahoo's adjusted prices shift by about 1e-6 between downloads, which was enough to move the stock ML result between $484 and $965).
- **Live paper:** from 2026-10-01, one day at a time, by GitHub Actions every evening. Only this record is real evidence, and the live record is never recomputed. One known deviation: the crypto accounts were set up before Sep 30 had closed, so their first fill landed on Sep 30, a day early. Fixed on 2026-10-04; the recorded days stay as they happened.

## Self-tuning

Each engine has a fixed menu of settings (`paper_trader/variants.py`): 38 for the ML model (an entry line from 52% to 62%, an exit line up to 7 points below it, a minimum hold of 1, 5 or 10 days, the original re-rank rule, and sitting in cash) and 37 for the tournament (the original strategies plus slower and faster versions, a 21 to 252 day lookback, monthly, weekly or daily re-weighting, scored after costs). Every entry runs as a shadow account on the same prices. Each night the layer (`paper_trader/selftune.py`) scores every entry on its trailing 126 trading days after costs and moves to the leader only when it is ahead of the current choice by 2% (log return). The window, the margin and the menus were written down before any result was seen.

The layer is a function of price history alone, so the truncation tests cover it: run it on data cut off at day *t* and the choice for *t* does not change. Each change the live accounts make is logged in `results/live/settings.csv` with its reason and shown on the page; the live record itself is never recomputed.

## Findings so far (backtest)

- **Self-tuning, 2018 to Sep 2026** (`results/backtest/summary.json`, the `_fixed` accounts are the original rules on the same prices): ML stocks $597 fixed → $2,256 tuned; ML crypto $2.75 → $9,761; tournament stocks $1,255 → $1,497; tournament crypto $250 → $2,671. Most of the gain is fees saved by trading less (the fixed ML accounts turned over half the account a day; the tuned ones hold for days). On the history the layer switched settings every 3 to 11 weeks.
- **What that does not show** (`scripts/controls.py`, `results/backtest/controls.json`): none of the four beat holding its own universe at equal weight from day one ($2,348 stocks, $7,864 crypto; the universe is today's winners). A no-skill control, the model's odds shuffled across names each day and run through the same menu and layer five times, ends at $1,032 to $2,391 on stocks and $735 to $2,261 on crypto: the stock result sits inside that range. The result also moves a lot with the layer's own window and margin (a 63-day window did worst on every account), which marks it as fragile. The menu was chosen knowing fees were the problem. It is still one backtest on a survivor universe.

- The tournament nearly matched the S&P 500 on stocks ($1,255 vs $1,300) with a smaller worst drop (-25% vs -34%) and a higher Sharpe ratio (0.91 vs 0.81). None of that is evidence of skill: the daily return difference from SPY has a t-statistic of -0.27, each Sharpe ratio carries a standard error of about 0.40 over this sample, and the smaller drop mostly reflects a beta of 0.69 to the market. It is one sample period, and the strategy thresholds were never tested out of sample.
- The ML model's daily up/down calls on stocks are right 53.2% of the time, slightly worse than always guessing "up" (stocks rose on 53.5% of the scored days). It turns over 57% of the stock account a day. Fees decide its result: $2,083 with no fees, $597 at 0.05%, $171 at 0.1%. In crypto, 0.55% per trade took it from $3,620 (no fees) to $2.75.
- Survivorship bias: the universe is today's large names, which flatters every backtest here.

## Code map

| File | What it does |
|---|---|
| `paper_trader/strategies.py` | Momentum, mean reversion and trend, each a function from prices to target weights, with speed parameters |
| `paper_trader/tournament.py` | Capital allocation across the three strategies by trailing Sharpe, at a chosen lookback and cadence |
| `paper_trader/features.py`, `ml.py` | Nine price features, the forward label, weekly walk-forward retraining |
| `paper_trader/variants.py`, `selftune.py` | The menus of settings per engine, and the layer that follows the leader after costs |
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
.venv/bin/python scripts/backtest.py     # about 3 minutes; uses the frozen prices; tuned and fixed accounts
.venv/bin/python scripts/controls.py     # the equal-weight bars and the shuffled-picks control
.venv/bin/python scripts/run_daily.py    # steps the live accounts and rebuilds site/
```
