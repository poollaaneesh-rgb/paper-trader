# Bloomberg check (BQuant)

Same stock tournament, same code, two data sources. The question: do the results hold up on Bloomberg's prices?

## In the FAR Lab

1. On a Terminal, type `BQNT <GO>` and start a new Python notebook. If BQNT isn't available on ASU's license, stop here and note it.
2. Paste all of `tournament_close.py` into one cell and run it. It needs about a minute.
3. Read the first lines. If it lists jumps over 35%, those are probably unadjusted stock splits. Write them down instead of trusting the numbers.
4. Copy the four numbers per row into the table below. Only summary numbers leave the Terminal. Bloomberg's license doesn't allow exporting the prices themselves.

If the BQL cell errors, the data call is the likely culprit (BQL syntax varies by version). The rest is plain pandas.

## Results

Simplified variant: closes only, fills at the next close, costs on weight turnover (drift ignored). Compare the two columns with each other, not with the main backtest.

| 2018-01-01 to 2026-09-30, $400 | Yahoo (frozen) | Bloomberg |
|---|---|---|
| Tournament: final / return / worst drop / Sharpe | $1,429 / +257.2% / -24.8% / 0.97 | |
| SPY hold: final / return / worst drop / Sharpe | $1,305 / +226.3% / -33.7% / 0.81 | |

Expect small differences: Bloomberg's PX_LAST is usually split-adjusted but not dividend-adjusted, while Yahoo's adjusted closes include dividends, so SPY hold should come out a little lower on Bloomberg.
