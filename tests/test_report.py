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
