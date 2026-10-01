# Paper Trader: design

Approved by Aneesh 2026-09-30 (tap answers in chat). Built with Claude Code.

## Purpose

A self-learning trading bot that trades **fake money only**, on real daily
market data, and publishes an honest scorecard. Two goals:

1. Find out, with evidence, whether automated strategies beat simply holding
   an index, for stocks and for crypto.
2. A public quant project for finance recruiting (code, method and results a
   recruiter can read).

## Out of scope (hard boundary)

- No brokerage connection of any kind (Schwab or otherwise), no live orders,
  no real money, no API keys for trading. The repo holds no secrets.
- No intraday trading. One decision per day.

## Accounts

Six fake accounts, each starting at **$400**:

| Account | Market | Engine |
|---|---|---|
| `tournament_stocks` | US stocks + ETFs | Strategy tournament |
| `tournament_crypto` | Crypto | Strategy tournament |
| `ml_stocks` | US stocks + ETFs | ML model |
| `ml_crypto` | Crypto | ML model |
| `bench_spy` | SPY | Buy and hold, never trades |
| `bench_btc` | BTC-USD | Buy and hold, never trades |

A bot account "wins" only if it beats its market's benchmark after costs.

## Universe

- **Stocks/ETFs (about 30):** SPY, QQQ, IWM, TLT, GLD, XLK, XLF, XLE, XLV,
  XLY, plus about 20 large US stocks (AAPL, MSFT, NVDA, AMZN, GOOGL, META,
  BRK-B, JPM, V, UNH, XOM, JNJ, PG, HD, COST, AVGO, LLY, WMT, MA, KO).
- **Crypto (6):** BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD, ADA-USD.
- Known limit, stated in the README: choosing today's large caps adds
  survivorship bias to the backtest.

## Engines

### Strategy tournament
Three sub-strategies, each with its own share of the account:
- **Momentum:** hold the top assets by 3-month return (skipping the last week).
- **Mean reversion:** buy assets with a 5-day RSI below 30, sell on recovery.
- **Trend:** hold assets whose price is above their 50-day average and whose
  50-day average is above their 200-day average.

On the first trading day of each month, capital is reweighted toward
sub-strategies in proportion to their trailing 3-month risk-adjusted return
(floor of 10% each so no strategy is ever switched off).

### ML model
- Features per asset per day (past data only): 1, 5, 20 and 60-day returns,
  20-day volatility, 14-day RSI, distance from 50 and 200-day averages,
  volume change.
- Target: whether tomorrow's open-to-open return is positive.
- Model: gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`),
  with logistic regression as a sanity baseline logged alongside.
- **Retrains weekly** (first trading day of each week; one setting,
  `RETRAIN_EVERY`), on an expanding window that ends the day before.
- Holds the top assets by predicted probability, only above 0.55, equal weight.

## Honest-testing rules

- **No look-ahead:** a decision on day *t* uses data up to the close of *t*;
  orders fill at the open of *t+1*. A test enforces this.
- **Costs on every trade:** stocks 0.05% slippage (no commission); crypto
  0.5% fee plus slippage.
- **Fractional shares** allowed (the account is $400).
- **Two phases, always labelled:**
  - *Backtest*: walk-forward from 2018-01-01 to 2026-09-30, so results exist
    on day one.
  - *Live paper*: from 2026-10-01, one real day at a time. Only this counts as
    evidence.

## Daily run (GitHub Actions)

Weekdays and weekends (crypto trades every day; stock accounts skip days the
market is closed), around 9 PM Arizona time:

1. Fetch daily prices (yfinance; on failure, Stooq for stocks and Coinbase's
   public candles for crypto). If every source fails, skip the day and record
   why. Never trade on stale data.
2. Fill yesterday's orders at today's open, then decide tomorrow's orders.
3. Retrain the ML model if due.
4. Append decisions to the trade log; write `results/state.json`,
   `results/trades.csv`, `results/equity.csv`, `results/summary.json`.
5. Rebuild the public page and commit the results back to the repo.

## Outputs

- **Public page** (GitHub Pages): equity curves against the benchmarks, with
  backtest and live paper visually separated. Per account: return, max
  drawdown, Sharpe, win rate, number of trades. The last 30 decisions with
  their reasons.
- **Trade log:** every decision with date, account, asset, action, size,
  fill price, cost and reason (the signal or probability behind it).
- **Life Dashboard card:** reads the public `summary.json`. Shows fake balances
  against the benchmarks, today's trades, and live-paper days so far.

## Testing

pytest, run in Actions before each daily run:
- the no-look-ahead check (shifting future prices must not change past
  decisions);
- cost accounting (a round trip loses exactly the modelled costs);
- portfolio arithmetic (cash plus positions equals equity, never negative cash);
- the benchmark accounts never trade;
- a data-source failure skips the day instead of trading.

## Stack

Python 3.11 (Actions), pandas, numpy, scikit-learn, yfinance, matplotlib or a
static Chart.js page. No paid services.
