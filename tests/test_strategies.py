import numpy as np
import pandas as pd
import pytest

from paper_trader import strategies, tournament
from tests.conftest import make_panel

FUNCS = [strategies.momentum, strategies.mean_reversion, strategies.trend]


@pytest.mark.parametrize("fn", FUNCS)
def test_no_lookahead(panel, fn):
    full = fn(panel, 2)
    for t in panel.dates[[250, 300, 399]]:
        part = fn(panel.truncate(t), 2)
        pd.testing.assert_series_equal(full.loc[t], part.loc[t], check_names=False)


@pytest.mark.parametrize("fn", FUNCS)
def test_weights_are_valid(panel, fn):
    w = fn(panel, 2)
    assert (w.fillna(0) >= 0).all().all()
    assert (w.fillna(0).sum(axis=1) <= 1 + 1e-9).all()


def test_tournament_no_lookahead(panel):
    full, _ = tournament.weights(panel, 2)
    for t in panel.dates[[250, 330, 399]]:
        part, _ = tournament.weights(panel.truncate(t), 2)
        pd.testing.assert_series_equal(full.loc[t], part.loc[t], check_names=False)


def test_allocations_floor_and_monthly(panel):
    w, alloc = tournament.weights(panel, 2)
    assert np.allclose(alloc.sum(axis=1), 1.0)
    assert (alloc >= 0.10 - 1e-12).all().all()
    months = pd.Series(alloc.index.to_period("M"), index=alloc.index)
    month_start = months != months.shift(1)
    changed = alloc.diff().abs().sum(axis=1) > 1e-12
    assert not (changed & ~month_start).any()
    assert (w.sum(axis=1) <= 1 + 1e-9).all()
