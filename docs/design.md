# Design

## Question

Do simple automated strategies beat holding an index after trading costs, in US stocks and in crypto? The project answers it with a backtest and, from 2026-10-01, a live paper record. Fake money only: no brokerage connection, no order API, no secrets in the repo.

## Accounts

Six accounts, each starting at $400. A bot "wins" only if it beats its market's buy-and-hold benchmark after costs.

| Account | Universe | Engine |
|---|---|---|
| `tournament_stocks` / `tournament_crypto` | 30 US stocks and ETFs / 6 coins | Strategy tournament |
| `ml_stocks` / `ml_crypto` | same | Gradient-boosted trees |
| `bench_spy` / `bench_btc` | SPY / BTC | Buy and hold |

## Engines

Every engine is a function from price history to a table of target weights (dates x tickers). Keeping engines pure is what makes look-ahead testable: run the engine on data truncated at day *t* and the row for *t* must not change.

**Strategy tournament** (`strategies.py`, `tournament.py`)

- Momentum: the top names by 3-month return, skipping the latest week (the short-term reversal effect).
- Mean reversion: buy when 5-day RSI falls below 30, sell once it recovers above 50.
- Trend: hold names above their 50-day average while the 50-day is above the 200-day.
- On the first trading day of each month, capital moves toward the strategies with the best trailing 63-day Sharpe ratio. Each keeps a 10% floor so none is switched off by one bad quarter.

**ML model** (`features.py`, `ml.py`)

- Nine features per asset per day, all from data up to that close: 1, 5, 20 and 60-day returns, 20-day volatility, 14-day RSI, distance from the 50 and 200-day averages, volume change.
- Label: whether the open-to-open return from *t+1* to *t+2* is positive, which is exactly the holding period of a decision made at the close of *t*.
- scikit-learn `HistGradientBoostingClassifier`, retrained every week on an expanding window. The last training row is two trading days before the retrain date, because its label needs those two opens. A logistic regression is fitted alongside as a baseline and only reported.
- Holds up to the top N names whose predicted probability is above 0.55, equal weight.

## Simulation

- A decision at the close of *t* fills at the open of *t+1* (`simulate.py`).
- Costs on every trade: 0.05% slippage for stocks; 0.5% fee plus 0.05% slippage for crypto. Fractional units allowed.
- Rebalancing skips trades smaller than 1% of equity (and never under $1), so the account doesn't pay costs to chase rounding.
- A ticker with no usable price that day is left alone and valued at its last close.

## Backtest and live run

- Backtest: walk-forward, 2018-01-01 to 2026-09-30, with history from 2015 for warm-up. Prices are frozen in `results/backtest/` because Yahoo's adjusted prices drift by about 1e-6 between downloads, which was enough to move the ML result by hundreds of dollars.
- Live: GitHub Actions runs `scripts/run_daily.py` every evening. It fetches prices (yfinance, with Coinbase's public candles as the crypto fallback), drops today's still-forming candle, steps each account through any new complete days, and rebuilds the page. A market whose data is missing or stale is skipped for the day and the reason recorded; it never trades on stale data. Re-running on the same data changes nothing.

## Tests

`pytest` covers: no look-ahead in every engine (truncation), fills at the next open, a round trip losing exactly the modelled costs, cash never going negative, the benchmark trading once, the ML training cutoff, and a stale market being skipped. CI runs lint and tests on every push; the daily run also runs the tests before trading.

## Known limits

- **Survivorship bias:** the universe is today's large names, which flatters every backtest here.
- **Tournament scoring ignores costs:** strategies are ranked on gross returns, so a high-turnover strategy is slightly overrated.
- **Sharpe ratios use no risk-free rate**, and crypto annualises over 365 days, stocks over 252.
- **Strategy breadth:** mean reversion and trend can hold more than N names; weights shrink rather than the list being cut.
- **ML turnover:** daily re-ranking turns over about 57% of the stock account a day, and costs decide its result. A holding period or a hysteresis band is the obvious next experiment.
