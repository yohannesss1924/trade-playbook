"""Backtest The Trade Playbook rules on the current watchlist.

Replays every trading day of the last N years: each morning it runs the same
rules as screener.py on the bars through the prior close, picks up to six
trades diversified by sector, and simulates buy stop-limit entries, stops,
targets and the 10-day time exit with a cash account (no margin).

    python backtest.py --years 2 --capital 10000
"""
import argparse
import datetime as dt
import html
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yfinance as yf

import screener as S


def earnings_history(t):
    try:
        e = yf.Ticker(t).get_earnings_dates(limit=16)
        return t, sorted({x.date() for x in e.index})
    except Exception:
        return t, None


def run(args):
    src = args.file or S.newest_watchlist()
    watch = S.load_watchlist(src)
    tickers = list(watch)
    yf.set_tz_cache_location(str(S.ROOT / ".cache"))
    raw = yf.download(tickers + ["SPY"], period=f"{args.years + 1}y", interval="1d",
                      group_by="ticker", auto_adjust=False, progress=False, threads=True)
    bars = {}
    for t in tickers + ["SPY"]:
        try:
            d = raw[t].dropna(subset=["Open", "High", "Low", "Close"])
        except KeyError:
            continue
        if len(d):
            bars[t] = d
    with ThreadPoolExecutor(8) as ex:
        earn = dict(ex.map(earnings_history, [t for t in tickers if t in bars]))

    spy = bars.pop("SPY")
    start = spy.index[-1] - pd.DateOffset(years=args.years)
    days = [x for x in spy.index if x >= start]

    ind = {t: S.indicators(d) for t, d in bars.items()}
    pos_of = {t: {x: i for i, x in enumerate(d.index)} for t, d in bars.items()}
    sector = {t: watch[t]["sector"] or "Unknown" for t in bars}

    cash = float(args.capital)
    open_pos, trades, curve = {}, [], []
    counts = {"signals": 0, "not filled": 0, "gapped over limit": 0, "no cash": 0, "earnings": 0}

    def close_trade(p, day, px, why):
        nonlocal released
        released += px * p["shares"]
        p.update(exit_date=str(day.date()), exit=round(px, 2), status=why,
                 R=round((px - p["fill"]) / p["risk"], 2),
                 pnl=round((px - p["fill"]) * p["shares"], 2))
        trades.append(p)

    for day in days:
        released = 0.0  # exits free cash at the end of the day

        # 1. Manage open positions.
        for t in list(open_pos):
            i = pos_of[t].get(day)
            if i is None:
                continue
            p, b = open_pos[t], ind[t]
            p["held"] += 1
            if b["l"][i] <= p["stop"]:
                close_trade(p, day, min(b["o"][i], p["stop"]), "stop")
            elif b["h"][i] >= p["target"]:
                close_trade(p, day, max(b["o"][i], p["target"]), "target")
            elif p["held"] >= S.MAX_HOLD:
                close_trade(p, day, b["c"][i], "time exit")
            else:
                continue
            del open_pos[t]

        # 2. This morning's report, from bars through the prior close.
        cands = []
        for t in bars:
            n = pos_of[t].get(day)
            if n is None or n < 60 or t in open_pos:
                continue
            r = S.analyze_at(t, ind[t], n)
            if r["fail"]:
                continue
            ed = earn.get(t)
            if ed and any(0 <= (x - day.date()).days <= S.EARNINGS_LOOKAHEAD_DAYS for x in ed):
                counts["earnings"] += 1
                continue
            r["sector"] = sector[t]
            cands.append(r)
        cands.sort(key=lambda r: -r["R"])
        picks = S.diversify(cands, S.TOP_N)
        counts["signals"] += len(picks)

        # 3. Stop-limit orders for today's session.
        for r in picks:
            t, i, b = r["ticker"], pos_of[r["ticker"]][day], ind[r["ticker"]]
            if b["o"][i] > r["limit"]:
                counts["gapped over limit"] += 1
                continue
            if b["h"][i] < r["entry"]:
                counts["not filled"] += 1
                continue
            fill = max(float(b["o"][i]), r["entry"])
            cost = fill * r["shares"]
            if cost > cash or r["shares"] == 0:
                counts["no cash"] += 1
                continue
            cash -= cost
            p = dict(ticker=t, sector=r["sector"], signal_date=str(day.date()), entry=r["entry"],
                     limit=r["limit"], stop=r["stop"], target=r["target"], shares=r["shares"],
                     risk=r["risk"], fill=round(fill, 2), fill_date=str(day.date()), held=1)
            # Fill day: stop checked before target.
            if b["l"][i] <= p["stop"]:
                close_trade(p, day, p["stop"], "stop")
            elif b["h"][i] >= p["target"]:
                close_trade(p, day, p["target"], "target")
            else:
                open_pos[t] = p

        cash += released
        mtm = 0.0
        for t, p in open_pos.items():
            i = pos_of[t].get(day)
            last = ind[t]["c"][i] if i is not None else p["fill"]
            mtm += last * p["shares"]
        curve.append((day, cash + mtm, len(open_pos)))

    eq = pd.Series([e for _, e, _ in curve], index=[d for d, _, _ in curve])
    spy_eq = args.capital * spy["Close"].loc[eq.index] / spy["Close"].loc[eq.index[0]]
    return dict(src=src, tickers=len(tickers), have=len(bars), earn_ok=sum(1 for v in earn.values() if v),
                trades=pd.DataFrame(trades), open_pos=open_pos, eq=eq, spy=spy_eq,
                counts=counts, start=eq.index[0], end=eq.index[-1], capital=args.capital,
                peak_open=max(n for _, _, n in curve))


def drawdown(s):
    return float((s / s.cummax() - 1).min())


def svg_curve(eq, spy, w=760, h=220):
    lo, hi = min(eq.min(), spy.min()), max(eq.max(), spy.max())
    pad = (hi - lo) * 0.05 or 1
    lo, hi = lo - pad, hi + pad

    def pts(s):
        n = len(s) - 1
        return " ".join(f"{i / n * w:.1f},{h - (v - lo) / (hi - lo) * h:.1f}" for i, v in enumerate(s))
    base = h - (eq.iloc[0] - lo) / (hi - lo) * h
    return (f'<svg viewBox="0 0 {w} {h + 20}" width="100%" style="max-width:{w}px">'
            f'<line x1="0" x2="{w}" y1="{base:.1f}" y2="{base:.1f}" stroke="#bbb" stroke-dasharray="4 4"/>'
            f'<polyline fill="none" stroke="#999" stroke-width="1.5" points="{pts(spy)}"/>'
            f'<polyline fill="none" stroke="#1f6feb" stroke-width="2" points="{pts(eq)}"/>'
            f'<text x="0" y="{h + 16}" font-size="12" fill="#666">{eq.index[0]:%b %Y}</text>'
            f'<text x="{w}" y="{h + 16}" font-size="12" fill="#666" text-anchor="end">{eq.index[-1]:%b %Y}</text>'
            f'</svg><p><small><span style="color:#1f6feb">&#9644;</span> Playbook &nbsp; '
            f'<span style="color:#999">&#9644;</span> SPY buy and hold, same $ start &nbsp; dashed = starting balance</small></p>')


def report(res):
    T, eq, spy, cap = res["trades"], res["eq"], res["spy"], res["capital"]
    end_val = float(eq.iloc[-1])
    n = len(T)
    wins = T[T["R"] > 0] if n else T
    losses = T[T["R"] <= 0] if n else T
    pf = (wins["pnl"].sum() / -losses["pnl"].sum()) if n and losses["pnl"].sum() < 0 else float("nan")
    by_exit = T["status"].value_counts().to_dict() if n else {}

    stats = [
        ("Period", f"{res['start']:%b %d, %Y} to {res['end']:%b %d, %Y}"),
        ("Starting balance", S.money(cap)),
        ("Ending balance", f"{S.money(end_val)} ({end_val / cap - 1:+.1%})"),
        ("SPY buy and hold, same period", f"{S.money(float(spy.iloc[-1]))} ({float(spy.iloc[-1]) / cap - 1:+.1%})"),
        ("Worst drop from a peak", f"{drawdown(eq):.1%} (SPY {drawdown(spy):.1%})"),
        ("Trades taken", f"{n}"),
        ("Win rate", f"{len(wins) / n:.0%}" if n else "n/a"),
        ("Average trade", f"{T['R'].mean():+.2f}R ({S.money(T['pnl'].mean())})" if n else "n/a"),
        ("Average win / loss", f"{wins['R'].mean():+.2f}R / {losses['R'].mean():+.2f}R" if len(wins) and len(losses) else "n/a"),
        ("Profit factor", f"{pf:.2f}" if not math.isnan(pf) else "n/a"),
        ("Exits", ", ".join(f"{k} {v}" for k, v in by_exit.items()) or "n/a"),
        ("Picks that never became trades",
         f"{res['counts']['not filled']} not filled, {res['counts']['gapped over limit']} gapped over limit, "
         f"{res['counts']['no cash']} skipped for lack of cash"),
        ("Most positions open at once", f"{res['peak_open']}"),
    ]

    yearly = []
    if n:
        T["year"] = T["exit_date"].str[:4]
        for y, g in T.groupby("year"):
            yearly.append((y, len(g), f"{(g['R'] > 0).mean():.0%}", f"{g['R'].sum():+.1f}R", S.money(g["pnl"].sum())))
    monthly = eq.resample("ME").last()
    mret = monthly.pct_change()
    mret.iloc[0] = monthly.iloc[0] / cap - 1

    sect = []
    if n:
        for s_, g in T.groupby("sector"):
            sect.append((s_, len(g), f"{(g['R'] > 0).mean():.0%}", f"{g['R'].sum():+.1f}R"))
        sect.sort(key=lambda x: -x[1])

    caveats = [
        "The watchlist is today's list. Those stocks were chosen because they did well recently, so the "
        "past two years favour them (look-ahead bias). Real results with each week's fresh list would likely be worse.",
        "Fills are simulated from daily bars: a day that touches both stop and target counts as a stop; "
        "gaps through the stop exit at the open. No commissions or slippage are included.",
        (f"Earnings dates were found for all {res['have']} stocks, so the earnings rule was applied throughout."
         if res["earn_ok"] == res["have"] else
         f"Earnings dates were found for {res['earn_ok']} of {res['have']} stocks; the others had no earnings filter."),
        "Risk is a fixed $100 per trade, as the playbook says, so gains are not compounded.",
    ]

    md = [f"# Trade Playbook backtest: {res['start']:%b %Y} to {res['end']:%b %Y}",
          f"Rules from screener.py applied to {res['src'].name} ({res['tickers']} stocks), "
          f"$10,000 paper account, $100 risk per trade, up to 6 picks a day.", ""]
    md += [f"- **{k}:** {v}" for k, v in stats]
    md += ["", "## By year (by exit date)", "", "| Year | Trades | Win rate | Total R | P&L |", "|---|---|---|---|---|"]
    md += [f"| {' | '.join(map(str, y))} |" for y in yearly]
    md += ["", "## By sector", "", "| Sector | Trades | Win rate | Total R |", "|---|---|---|---|"]
    md += [f"| {' | '.join(map(str, x))} |" for x in sect]
    md += ["", "## Monthly balance", "", "| Month | Balance | Change |", "|---|---|---|"]
    md += [f"| {d:%b %Y} | {S.money(v)} | {r:+.1%} |" for d, v, r in zip(monthly.index, monthly, mret)]
    md += ["", "## Read this before trusting the numbers", ""] + [f"- {c}" for c in caveats]
    text = "\n".join(md)

    def table(head, rows):
        return ("<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" +
                "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows) +
                "</table>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Playbook backtest</title>
<style>body{{font:14px/1.5 -apple-system,Segoe UI,Arial,sans-serif;color:#1d1d1f;max-width:860px;margin:24px auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%;margin:8px 0 20px}}th,td{{padding:5px 8px;border-bottom:1px solid #ddd;text-align:left}}
th{{background:#f4f4f5}}small{{color:#666}}</style></head><body>
<h2>Trade Playbook backtest: {res['start']:%b %Y} to {res['end']:%b %Y}</h2>
<p>{html.escape(md[1])}</p>
{svg_curve(eq, spy)}
{table(['Measure', 'Result'], stats)}
<h3>By year (by exit date)</h3>{table(['Year', 'Trades', 'Win rate', 'Total R', 'P&L'], yearly)}
<h3>By sector</h3>{table(['Sector', 'Trades', 'Win rate', 'Total R'], sect)}
<h3>Monthly balance</h3>{table(['Month', 'Balance', 'Change'], [(f'{d:%b %Y}', S.money(v), f'{r:+.1%}') for d, v, r in zip(monthly.index, monthly, mret)])}
<h3>Read this before trusting the numbers</h3><ul>{''.join(f'<li>{html.escape(c)}</li>' for c in caveats)}</ul>
</body></html>"""

    S.REPORTS.mkdir(exist_ok=True)
    stem = f"backtest-{dt.date.today():%Y-%m-%d}"
    (S.REPORTS / f"{stem}.md").write_text(text, encoding="utf-8")
    (S.REPORTS / f"{stem}.html").write_text(page, encoding="utf-8")
    if n:
        T.drop(columns=["held", "year"], errors="ignore").to_csv(S.REPORTS / f"{stem}-trades.csv", index=False)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--capital", type=float, default=10000)
    ap.add_argument("--file", type=Path)
    print(report(run(ap.parse_args())))
