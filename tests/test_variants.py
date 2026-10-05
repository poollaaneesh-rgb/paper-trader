import numpy as np
import pandas as pd

from paper_trader import variants

IDX = pd.date_range("2024-01-01", periods=8, freq="B")


def test_band_variant_enters_holds_through_the_minimum_and_exits_below_the_exit_line():
    probs = pd.DataFrame({"AAA": [0.5, 0.6, 0.56, 0.52, 0.52, 0.6, 0.5, 0.5]}, index=IDX)
    w = variants.ml_band(probs, 1, th_in=0.58, th_out=0.55, min_hold=3)
    assert w["AAA"].tolist() == [0, 1, 1, 1, 0, 1, 1, 1]


def test_fixed_ml_variant_is_the_old_top_n_rule():
    probs = pd.DataFrame({"AAA": [0.6, 0.56], "BBB": [0.7, 0.5], "CCC": [0.56, 0.5]}, index=IDX[:2])
    w = variants.ml_variants(probs, 2)[variants.ML_FIXED]
    assert w.iloc[0].tolist() == [0.5, 0.5, 0.0]
    assert w.iloc[1].tolist() == [0.5, 0.0, 0.0]


def test_ml_menu_has_the_fixed_rule_cash_and_a_description_for_every_entry():
    probs = pd.DataFrame({"AAA": np.linspace(0.4, 0.7, 8), "BBB": np.linspace(0.7, 0.4, 8)}, index=IDX)
    menu = variants.ml_variants(probs, 1)
    assert variants.ML_FIXED in menu and variants.ML_CASH in menu
    assert len(menu) == 38
    assert (menu[variants.ML_CASH] == 0).all().all()
    for vid in menu:
        text = variants.describe(vid)
        assert text and vid not in text
    assert variants.describe("ml:in58-out51-hold10") == "enter above 58% odds, exit below 51%, hold at least 10 days"


def test_tournament_menu_is_valid_and_described(panel):
    menu = variants.tournament_variants(panel, 2, 0.005)
    assert variants.TOUR_FIXED in menu
    assert len(menu) == 37
    for vid, w in menu.items():
        assert (w.fillna(0) >= 0).all().all()
        assert (w.fillna(0).sum(axis=1) <= 1 + 1e-9).all()
        assert variants.describe(vid)
    assert variants.describe("tour:slow-126-W-net") == (
        "slower strategies, capital re-weighted weekly by the last 126 days after costs"
    )
    assert variants.describe(variants.TOUR_FIXED) == (
        "the original rules: capital re-weighted monthly by the last 63 days before costs"
    )
