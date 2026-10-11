# Trade Playbook

Morning report for The Trade Playbook: the four checks and the building-the-trade
rules, applied to the weekly watchlist.

## Every Sunday

Upload the new watchlist (Finviz CSV or Excel) into `inbox/` on GitHub:
open the `inbox` folder → **Add file** → **Upload files** → drop the file → **Commit changes**.
The newest file in `inbox/` is the one used, all week.

## Files

- `screener.py` — the daily report. Writes `reports/<date>.html|.md` and updates `journal.csv`.
- `backtest.py` — replays the rules over past years: `python backtest.py --years 2 --capital 10000`.
- `journal.csv` — every pick and how it played out (paper results from daily bars).
