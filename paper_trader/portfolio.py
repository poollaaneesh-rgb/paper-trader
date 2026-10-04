"""A cash-and-units account with proportional trading costs. Fake money only."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Trade:
    ticker: str
    units: float  # positive buy, negative sell
    price: float
    cost: float


def _ok(price) -> bool:
    return price is not None and not (isinstance(price, float) and math.isnan(price)) and price > 0


@dataclass
class Account:
    cash: float
    positions: dict = field(default_factory=dict)

    def equity(self, prices: pd.Series) -> float:
        total = self.cash
        for t, units in self.positions.items():
            p = prices.get(t)
            if _ok(p):
                total += units * p
        return total

    def rebalance(
        self, weights: pd.Series, prices: pd.Series, cost_rate: float, value_prices: pd.Series | None = None
    ) -> list[Trade]:
        """Trade toward target weights at `prices`. Tickers without a usable price are left alone.

        `value_prices` (last known closes) values untradable holdings so the targets use true equity.
        """
        weights = weights.fillna(0.0)
        valuation = prices if value_prices is None else prices.combine_first(value_prices)
        eq = self.equity(valuation)
        threshold = max(1.0, 0.01 * eq)
        names = set(weights.index[weights > 0]) | set(self.positions)
        sells, buys = {}, {}
        for t in names:
            p = prices.get(t)
            if not _ok(p):
                continue
            current = self.positions.get(t, 0.0) * p
            target = float(weights.get(t, 0.0)) * eq / (1 + cost_rate)
            diff = target - current
            full_exit = target == 0 and current > 0
            if abs(diff) < threshold and not full_exit:
                continue
            (sells if diff < 0 else buys)[t] = diff
        trades = []
        for t, diff in sells.items():
            p = prices[t]
            units = self.positions[t] if weights.get(t, 0.0) == 0 else -diff / p
            value = units * p
            cost = value * cost_rate
            self.cash += value - cost
            left = self.positions[t] - units
            if left <= 1e-12:
                del self.positions[t]
            else:
                self.positions[t] = left
            trades.append(Trade(t, -units, p, cost))
        need = sum(buys.values()) * (1 + cost_rate)
        scale = min(1.0, self.cash / need) if need > 0 else 1.0
        for t, diff in buys.items():
            p = prices[t]
            value = diff * scale
            if value < threshold:  # cash ran short; a sliver of a buy is not worth logging or paying for
                continue
            cost = value * cost_rate
            self.cash -= value + cost
            self.positions[t] = self.positions.get(t, 0.0) + value / p
            trades.append(Trade(t, value / p, p, cost))
        if -1e-9 < self.cash < 0:
            self.cash = 0.0
        return trades
