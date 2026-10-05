import pandas as pd

from paper_trader import engines, tournament, variants
from paper_trader.config import Market
from tests.conftest import make_panel

MKT = Market("crypto", ("AAA", "BBB", "CCC", "DDD"), "AAA", 0.005, 2)
LONG = make_panel(n_days=760)


def test_tuned_tournament_has_no_lookahead(panel):
    start = panel.dates[300]
    full, _, diag = engines.targets("tournament", MKT, panel, start)
    assert diag["chosen"].index[-1] == panel.dates[-1]
    for t in panel.dates[[320, 360, 399]]:
        part, _, _ = engines.targets("tournament", MKT, panel.truncate(t), start)
        pd.testing.assert_series_equal(full.loc[t], part.loc[t], check_names=False)


def test_tuned_tournament_exposes_the_original_rule_for_comparison(panel):
    _, _, diag = engines.targets("tournament", MKT, panel, panel.dates[300])
    pd.testing.assert_frame_equal(diag["fixed_weights"], tournament.weights(panel, 2)[0])
    assert diag["variant"] in variants.describe(diag["variant"]) or variants.describe(diag["variant"])


def test_tuned_ml_reports_its_choice_and_explains_trades():
    start = LONG.dates[600]
    w, explain, diag = engines.targets("ml", MKT, LONG, start)
    assert (w.fillna(0).sum(axis=1) <= 1 + 1e-9).all()
    chosen = diag["chosen"]
    assert chosen.index[0] < start <= chosen.index[-1]
    assert set(chosen.unique()) <= set(
        variants.ml_variants(pd.DataFrame(index=LONG.dates[:1], columns=LONG.tickers, dtype=float), 2)
    )
    d = LONG.dates[650]
    text = explain(d, "AAA", "buy")
    assert "odds" in text and variants.describe(chosen.loc[d]) in text
    assert "fixed_weights" in diag and diag["fixed_weights"].shape[1] == 4
