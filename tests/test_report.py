import json
import re
from pathlib import Path

from paper_trader import engines, report

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_every_account_is_labelled_and_listed_once_on_the_page():
    listed = [name for names in report.MARKET_ACCOUNTS.values() for name in names]
    assert sorted(listed) == sorted(engines.ACCOUNTS)
    assert set(engines.ACCOUNTS) <= set(report.LABELS)


def test_the_page_reads_each_market_as_tournament_then_ml_then_the_benchmarks():
    # page_template.html takes the first entry as the tournament, the second as the ML model and the third as the
    # index to beat, so the equal-weight benchmark has to come after them.
    for market, names in report.MARKET_ACCOUNTS.items():
        assert [engines.ACCOUNTS[n][1] for n in names] == ["tournament", "ml", "bench", "bench_eq"]
        assert {engines.ACCOUNTS[n][0] for n in names} == {market}


def test_page_carries_a_live_curve_for_every_live_account(tmp_path):
    report.build(RESULTS, tmp_path)
    html = (tmp_path / "index.html").read_text()
    assert 'id="tiles"' in html
    data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1))
    live = data["summary"]["live"]["accounts"]
    started = [n for n, a in live.items() if a.get("days")]
    assert started
    for name in started:
        curve = data["live_curves"][name]
        assert len(curve["dates"]) == len(curve["equity"]) > 0
