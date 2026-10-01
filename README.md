# Paper Trader

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
- **Live paper:** from 2026-10-01, one day at a time, by GitHub Actions every evening. Only this record is real evidence.

## Findings so far (backtest)

- The tournament nearly matched the S&P 500 on stocks ($1,255 vs $1,300) with a smaller worst drop (-25% vs -34%) and a higher Sharpe ratio (0.91 vs 0.81).
- The ML model's daily up/down calls are right about 53% of the time, close to the base rate, and it turns over 57% of the stock account a day. Fees decide its result: $2,083 with no fees, $597 at 0.05%, $171 at 0.1%. In crypto, 0.55% per trade took it from $3,620 (no fees) to $2.75.
- Survivorship bias: the universe is today's large names, which flatters every backtest here.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python scripts/backtest.py     # about 2.5 minutes; uses the frozen prices
.venv/bin/python scripts/run_daily.py    # steps the live accounts and rebuilds site/
```

Built with Claude Code.
