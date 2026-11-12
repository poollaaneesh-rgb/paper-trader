import json

from paper_trader import engines, report


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


def test_the_chart_draws_every_date_of_every_series():
    # The x axis used to be the last series' dates. Once a benchmark started a day after the bots (the crypto
    # bots have a Sep 30 point), that day fell off the chart, so the axis must be the sorted union of all series.
    assert "[...new Set(series.flatMap(n => curves[n].dates))].sort()" in report.TEMPLATE
    assert "curves[series[series.length - 1]].dates" not in report.TEMPLATE


def test_the_feed_carries_what_the_weekly_seats_read(tmp_path):
    account = {"final_equity": 410.0, "total_return": 0.025, "days": 31, "n_trades": 4, "sharpe": 1.0}
    seat_fields = {"return_30d": 0.03, "bankrupt": None, "targets": {"BTC-USD": 0.5}}
    summary = {
        "generated_at": "2026-11-11T11:55+00:00",
        "live": {"start": "2026-10-01", "accounts": {"ml_crypto": {**account, **seat_fields}}},
        "backtest": {"accounts": {"ml_crypto": {"final_equity": 9760.99, "total_return": 23.4}}},
    }
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "summary.json").write_text(json.dumps(summary))
    report.build(tmp_path / "results", tmp_path / "site")
    feed = json.loads((tmp_path / "site" / "summary.json").read_text())
    assert feed["live"]["ml_crypto"] == {
        "final_equity": 410.0,
        "total_return": 0.025,
        "days": 31,
        "n_trades": 4,
        **seat_fields,
    }
