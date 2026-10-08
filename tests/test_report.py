import json
import re
from pathlib import Path

from paper_trader import report

RESULTS = Path(__file__).resolve().parents[1] / "results"


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
