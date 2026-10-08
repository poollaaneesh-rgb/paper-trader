"""Universes, costs and dates. Fake money only: nothing here talks to a broker."""

from dataclasses import dataclass

START_CASH = 400.0
DATA_START = "2015-01-01"  # warm-up history for 200-day averages and the first ML fits
BACKTEST_START = "2018-01-01"
BACKTEST_END = "2026-09-30"
LIVE_START = "2026-10-01"
# Every trading account runs on its own until it is bankrupt (the user's decision, 2026-10-08): once its equity is
# under the broker's smallest order it sells what is left and never trades again. Its record stays on the site.
BANKRUPT_AT = 1.0
RETRAIN_EVERY = "W"  # ML retrains on the first trading day of each week
ML_THRESHOLD = 0.55
TOURNAMENT_FLOOR = 0.10
SLIPPAGE = 0.0005


@dataclass(frozen=True)
class Market:
    name: str
    tickers: tuple[str, ...]
    benchmark: str
    cost_rate: float
    max_positions: int


STOCKS = Market(
    name="stocks",
    tickers=(
        "SPY",
        "QQQ",
        "IWM",
        "TLT",
        "GLD",
        "XLK",
        "XLF",
        "XLE",
        "XLV",
        "XLY",
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "BRK-B",
        "JPM",
        "V",
        "UNH",
        "XOM",
        "JNJ",
        "PG",
        "HD",
        "COST",
        "AVGO",
        "LLY",
        "WMT",
        "MA",
        "KO",
    ),
    benchmark="SPY",
    cost_rate=SLIPPAGE,
    max_positions=5,
)

# Crypto cost a side: Alpaca's tier-one taker fee plus slippage, from 2026-10-07 (0.005 + SLIPPAGE before). The
# self-tuning layer re-scores every setting at the current cost, so a switch it makes because of the change says so.
FEE_CHANGE_NOTE = (
    "re-scored under the crypto fee changed on 2026-10-07 (0.25% + 0.05% a side); the layer's choice moved with it"
)

CRYPTO = Market(
    name="crypto",
    tickers=("BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD"),
    benchmark="BTC-USD",
    cost_rate=0.0025 + SLIPPAGE,
    max_positions=3,
)

MARKETS = {"stocks": STOCKS, "crypto": CRYPTO}
