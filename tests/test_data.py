import pandas as pd

from paper_trader.data import complete_days
from tests.conftest import make_panel


def test_complete_days_drops_today_and_later():
    p = make_panel(n_days=10, freq="D")
    today = p.close.index[7]
    out = complete_days(p, today)
    assert out.close.index.max() == p.close.index[6]
    assert len(out.open) == 7
