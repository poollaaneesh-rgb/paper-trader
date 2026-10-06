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

## Self-tuning layer (2026-10-05)

Each engine's rules are one point in a small menu of settings (`variants.py`), and every menu entry is a full table of target weights, so it can be scored as a shadow account on the same prices. The layer (`selftune.py`) scores every entry on its trailing WINDOW (126 trading days) return after costs, where the daily return of weights decided at close *t* is their open *t+1* to open *t+2* return less the cost rate times the turnover from the previous row. It moves to the leader only when the leader is ahead of the current choice by MARGIN (2%, log return over the window). The review runs every night; on the 2018 to 2026 prices it changed settings every 3 to 11 weeks.

- **ML menu (38):** the original re-rank rule (top N above 55% odds, re-ranked daily); an entry line in {52, 55, 58, 62}% with an exit line 0, 3 or 7 points below it and a minimum hold of 1, 5 or 10 days; and cash. The band rule enters when the odds pass the entry line and keeps the name until the odds fall below the exit line and the hold is served.
- **Tournament menu (37):** the original rules (monthly, 63-day lookback, gross scoring); and the original, slower and faster strategy settings crossed with a 21, 63, 126 or 252 day lookback and monthly, weekly or daily re-weighting, scored net of each strategy's own costs.
- **Purity:** both the variants and the layer are functions of price history, so the truncation tests apply to the self-tuned engines as they did to the fixed ones. Live, the ML odds are computed from WARMUP (252 trading days) before the live start so the first live choice rests on a real shadow record; the first live choice is therefore path-dependent and may differ from the backtest's choice on the same date.
- **Record:** `results/live/settings.csv` holds every change (date, account, variant, description, reason). The live equity and trade records are never recomputed.
- **Backtest and controls:** `scripts/backtest.py` simulates each tuned account and, from the same menu, its original fixed rules (`*_fixed`). `scripts/controls.py` adds the bars a tuned result must clear: the universe held at equal weight from day one, and the model's odds shuffled across names each day through the same menu and layer (five runs).
- **Why the three numbers are fixed:** the window, margin and menus were set before any result was seen, so the layer itself is not tuned on the test period. Changing them after seeing results would turn the backtest into a fit.

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
- **ML turnover:** the original daily re-ranking turned over about 57% of the stock account a day, and costs decided its result; the self-tuning layer's band and hold entries are that experiment, and most of their backtest gain is the fees they save.
- **Self-tuning is fragile:** the result moves a lot with the layer's own window and margin, none of the tuned accounts beat holding its universe at equal weight, and a shuffled-picks control reproduces much of the stock result. The live record, with its change log, is the only evidence that counts.
