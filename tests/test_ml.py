import pandas as pd

from paper_trader import ml
from tests.conftest import make_panel

# Features need 200 days and training needs 252 more, so ML tests use a longer panel.
LONG = make_panel(n_days=700)


def test_no_lookahead():
    panel = LONG
    start = panel.dates[470]
    full, _ = ml.weights(panel, 2, start=start, threshold=0.5)
    assert full.to_numpy().sum() > 0
    for t in panel.dates[[500, 600, 699]]:
        part, _ = ml.weights(panel.truncate(t), 2, start=start, threshold=0.5)
        pd.testing.assert_series_equal(full.loc[t], part.loc[t], check_names=False)


def test_training_cutoff_is_two_days_before_boundary(panel):
    dates = panel.dates
    b = dates[300]
    assert ml.train_end(dates, b) == dates[298]


def test_holds_nothing_above_impossible_threshold(panel):
    w, _ = ml.weights(panel, 2, start=panel.dates[280], threshold=1.0)
    assert (w.fillna(0) == 0).all().all()


def test_produces_some_positions():
    panel = LONG
    w, diag = ml.weights(panel, 2, start=panel.dates[470], threshold=0.0)
    assert w.loc[panel.dates[600]].sum() > 0
    assert diag["retrains"] > 0
