"""Morning report for The Trade Playbook.

Reads the newest watchlist in inbox/, runs the four checks and the
building-the-trade rules on daily bars from Yahoo Finance, and writes
reports/<date>.html, reports/<date>.md and reports/latest.json.

    python screener.py                 # normal run
    python screener.py --file list.xlsx
"""
import argparse
import datetime as dt
import html
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
INBOX = ROOT / "inbox"
REPORTS = ROOT / "reports"

RISK_DOLLARS = 100.0         # 1% of the account, fixed
TOP_N = 6
EARNINGS_LOOKAHEAD_DAYS = 15  # calendar days, covers the 3-10 trading day hold
PEAK_LOOKBACK = 15            # bars searched for the pullback's starting high
BASE_DAYS = 15                # three weeks sideways
LIMIT_ATR = 0.1               # stop-limit ceiling above the entry trigger
MAX_HOLD = 10                 # trading days before a time exit
JOURNAL = ROOT / "journal.csv"


# ---------------------------------------------------------------- watchlist

def newest_watchlist():
    files = [p for p in INBOX.iterdir()
             if p.suffix.lower() in (".xlsx", ".xlsm", ".xls", ".csv")
             and not p.name.startswith("~$")]
    if not files:
        sys.exit(f"No spreadsheet found in {INBOX}")
    return max(files, key=added_at)


def added_at(p):
    # In a git checkout every file has the checkout's mtime, so use the commit time.
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%ct", "--", p.name], cwd=p.parent,
                             capture_output=True, text=True, timeout=10).stdout.strip()
        if out:
            return float(out)
    except (OSError, subprocess.SubprocessError):
        pass
    return p.stat().st_mtime


def parse_date(x):
    d = pd.to_datetime(x, errors="coerce")
    return None if pd.isna(d) else d.date()


def load_watchlist(path):
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in lower:
                return lower[n]
        return None

    tcol = pick("ticker", "symbol", "tickers", "symbols", "stock", "ticker symbol")
    if tcol is None:
        tcol = df.columns[0]
    ncol = pick("company", "name", "company name", "description")
    scol = pick("sector")
    ecol = pick("earnings date", "earnings", "next earnings")

    rows = {}
    for _, r in df.iterrows():
        t = r[tcol]
        if pd.isna(t):
            continue
        t = str(t).strip().upper().replace(".", "-")
        if not t or " " in t or len(t) > 10:
            continue
        rows[t] = {
            "company": str(r[ncol]).strip() if ncol and not pd.isna(r[ncol]) else "",
            "sector": str(r[scol]).strip() if scol and not pd.isna(r[scol]) else "",
            "earnings": parse_date(r[ecol]) if ecol else None,
        }
    return rows


# ---------------------------------------------------------------- data

def download(tickers):
    yf.set_tz_cache_location(str(ROOT / ".cache"))
    data = yf.download(tickers, period="1y", interval="1d", group_by="ticker",
                       auto_adjust=False, progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            d = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            d = d.dropna(subset=["Open", "High", "Low", "Close"])
            if d.empty:
                raise KeyError(t)
        except KeyError:
            # Threaded downloads sometimes drop a ticker; retry it alone.
            d = yf.download(t, period="1y", interval="1d", auto_adjust=False,
                            progress=False, threads=False, multi_level_index=False)
            d = d.dropna(subset=["Open", "High", "Low", "Close"])
        # Before the close, drop any partial bar for today.
        now = dt.datetime.now()
        if len(d) and d.index[-1].date() == now.date() and now.hour < 16:
            d = d.iloc[:-1]
        if len(d):
            out[t] = d
    return out


def round_step(price):
    if price < 10:
        return 1
    if price < 50:
        return 5
    if price < 200:
        return 10
    if price < 1000:
        return 50
    return 100


# ---------------------------------------------------------------- the rules

def indicators(d):
    """Precompute everything the rules read, as plain arrays."""
    o, h, l, c, v = (d[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close", "Volume"))
    cs = d["Close"]
    ema20 = cs.ewm(span=20, adjust=False).mean().to_numpy()
    sma50 = cs.rolling(50).mean().to_numpy()
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - cs.shift()).abs(),
                    (d["Low"] - cs.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    vavg = d["Volume"].rolling(20).mean().to_numpy()
    return dict(o=o, h=h, l=l, c=c, v=v, ema20=ema20, sma50=sma50, atr=atr, vavg=vavg,
                dates=[x.date() for x in d.index])


def analyze(t, d):
    return analyze_at(t, indicators(d), len(d))


def analyze_at(t, ind, n):
    """Apply the four checks and the building rules using bars [0, n)."""
    r = {"ticker": t, "checks": {}, "fail": None}
    if n < 60:
        r["fail"] = "less than 60 days of history"
        return r

    o, h, l, c, v = ind["o"], ind["h"], ind["l"], ind["c"], ind["v"]
    ema20, sma50, atr, vavg = ind["ema20"], ind["sma50"], ind["atr"], ind["vavg"]
    A = float(atr[n - 1])
    C = float(c[n - 1])
    r.update(close=round(C, 2), atr=round(A, 2), asof=str(ind["dates"][n - 1]))

    # Pullback = bars after the most recent high of the last PEAK_LOOKBACK bars.
    win = h[n - PEAK_LOOKBACK:n]
    peak_pos = n - 1 - int(win[::-1].argmax())
    pb = slice(peak_pos + 1, n)
    pb_len = n - peak_pos - 1

    # Base variant: three weeks sideways with the swings narrowing.
    b0 = n - BASE_DAYS
    box = float(h[b0:n].max() - l[b0:n].min())
    early = float(h[b0:n - 7].max() - l[b0:n - 7].min())
    late = float(h[n - 7:n].max() - l[n - 7:n].min())
    drift = abs(C - float(c[b0]))
    is_base = box <= 4 * A and late < 0.75 * early and drift <= 1.5 * A

    # Check 1: within ~1 ATR of the line price actually came down to.
    c1 = False
    route, anchor_low, line_name, dist = None, None, None, None
    if pb_len >= 2:
        low_pos = peak_pos + 1 + int(l[pb].argmin())
        low = float(l[low_pos])
        lines = {"20 EMA": float(ema20[low_pos]), "50 SMA": float(sma50[low_pos])}
        line_name = min(lines, key=lambda k: abs(low - lines[k]))
        touched = abs(low - lines[line_name]) <= 0.5 * A
        now_line = float((ema20 if line_name == "20 EMA" else sma50)[n - 1])
        dist = (C - now_line) / A
        if touched and 0.3 <= dist <= 1.0:
            c1, route, anchor_low = True, "pullback", low
    if not c1 and is_base:
        c1, route, anchor_low, line_name, dist = True, "base", float(l[b0:n].min()), None, None
    r["checks"]["1 near the line"] = c1
    r["route"] = route
    r["line"] = line_name
    r["dist_atr"] = None if dist is None else round(dist, 2)

    # Check 2: daily ranges shrinking.
    rng_all = h - l
    rng_src = rng_all[pb] if route != "base" else rng_all[n - 4:n]
    rng = [float(x) for x in rng_src[-4:]]
    ups = sum(1 for a, b in zip(rng, rng[1:]) if b > a)
    c2 = len(rng) >= 3 and rng[-1] < rng[0] and ups <= 1
    r["checks"]["2 ranges shrinking"] = c2
    r["ranges"] = [round(x, 2) for x in rng]

    # Check 3: volume below average on the way down (base satisfies it).
    if route == "base":
        c3 = True
    elif pb_len >= 1:
        ref = float(vavg[peak_pos]) if not math.isnan(vavg[peak_pos]) else float(vavg[n - 1])
        c3 = float(v[pb].mean()) < ref
    else:
        c3 = False
    r["checks"]["3 volume below avg"] = c3

    # Check 4: no gap-down below a moving average in the last 10 days.
    gap = False
    for i in range(n - 10, n):
        if o[i] < l[i - 1] and (o[i] < ema20[i - 1] or o[i] < sma50[i - 1]):
            gap = True
            break
    r["checks"]["4 no gap-down"] = not gap

    failed = [k for k, ok in r["checks"].items() if not ok]
    if failed:
        r["fail"] = "failed check " + ", ".join(k.split()[0] for k in failed)
        return r

    # Building the trade (earnings is checked later, for survivors only).
    prior_high = float(h[n - 1])
    entry = round(prior_high + max(0.02, 0.001 * prior_high), 2)
    stop = round(anchor_low - 0.25 * A, 2)
    risk = round(entry - stop, 2)
    risk_atr = risk / A
    step = round_step(entry)
    rn = math.ceil(entry / step) * step
    target = round(rn - max(0.05, 0.001 * rn), 2)
    if target <= entry:
        rn += step
        target = round(rn - max(0.05, 0.001 * rn), 2)
    R = (target - entry) / risk if risk > 0 else 0
    limit = round(entry + max(0.05, LIMIT_ATR * A), 2)
    r.update(entry=entry, limit=limit, stop=stop, risk=risk, risk_atr=round(risk_atr, 2),
             round_number=rn, target=target, R=round(R, 2),
             shares=int(RISK_DOLLARS // risk) if risk > 0 else 0)

    if not 1.0 <= risk_atr <= 1.5:
        r["fail"] = f"risk is {risk_atr:.2f} ATR (needs 1.0-1.5)"
    elif R < 2.0:
        r["fail"] = f"R is {R:.2f} to the ${rn:g} round number (needs 2.0)"
    return r


def next_earnings(t):
    try:
        cal = yf.Ticker(t).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            future = [x for x in dates if x >= dt.date.today()]
            return min(future) if future else None, True
        return None, True
    except Exception:
        return None, False


def sector_of(t, given):
    if given:
        return given
    try:
        return yf.Ticker(t).info.get("sector") or "Unknown"
    except Exception:
        return "Unknown"


def diversify(cands, n):
    picked, seen = [], set()
    for c in cands:
        if c["sector"] not in seen:
            picked.append(c)
            seen.add(c["sector"])
        if len(picked) == n:
            return picked
    for c in cands:
        if c not in picked:
            picked.append(c)
        if len(picked) == n:
            break
    return picked


# ---------------------------------------------------------------- tracking
#
# Every pick is logged to journal.csv and followed as if the stop-limit order
# were placed for the session on or after the report date and cancelled at that
# day's close if unfilled. Once filled it exits at the stop, the target or the
# close of day MAX_HOLD. When a day touches both stop and target, the stop is
# assumed to come first.

JCOLS = ["signal_date", "ticker", "sector", "entry", "limit", "stop", "target", "shares",
         "risk", "status", "fill_date", "fill", "exit_date", "exit", "R", "pnl"]
CLOSED = ("target", "stop", "time exit")


def load_journal():
    if JOURNAL.exists():
        return pd.read_csv(JOURNAL, dtype={"signal_date": str, "fill_date": str, "exit_date": str})
    return pd.DataFrame(columns=JCOLS)


def settle(j, bars):
    for i, s in j.iterrows():
        if s["status"] not in ("pending", "open") or s["ticker"] not in bars:
            continue
        d = bars[s["ticker"]]
        after = d[d.index.date >= dt.date.fromisoformat(s["signal_date"])]
        if s["status"] == "pending":
            if after.empty:
                continue
            day = after.iloc[0]
            if day["Open"] > s["limit"]:
                j.at[i, "status"] = "gapped over limit"
                continue
            if day["High"] < s["entry"]:
                j.at[i, "status"] = "not filled"
                continue
            j.at[i, "status"] = "open"
            j.at[i, "fill_date"] = str(after.index[0].date())
            j.at[i, "fill"] = round(max(float(day["Open"]), s["entry"]), 2)
        s = j.loc[i]
        held = after[after.index.date >= dt.date.fromisoformat(s["fill_date"])]
        for k, (ts, b) in enumerate(held.iterrows()):
            exit_px = None
            if b["Low"] <= s["stop"]:
                exit_px, status = (min(float(b["Open"]), s["stop"]) if k else s["stop"]), "stop"
            elif b["High"] >= s["target"]:
                exit_px, status = (max(float(b["Open"]), s["target"]) if k else s["target"]), "target"
            elif k + 1 >= MAX_HOLD:
                exit_px, status = float(b["Close"]), "time exit"
            if exit_px is not None:
                j.at[i, "status"] = status
                j.at[i, "exit_date"] = str(ts.date())
                j.at[i, "exit"] = round(exit_px, 2)
                j.at[i, "R"] = round((exit_px - s["fill"]) / s["risk"], 2)
                j.at[i, "pnl"] = round((exit_px - s["fill"]) * s["shares"], 2)
                break
    return j


def log_picks(j, picks, today):
    # A pick whose session hasn't traded yet is replaced by today's fresher one.
    j = j[j["status"] != "pending"]
    open_now = set(j.loc[j["status"] == "open", "ticker"])
    rows = []
    for p in picks:
        if p["ticker"] in open_now:
            p["already_open"] = True
            continue
        rows.append({"signal_date": str(today), "ticker": p["ticker"], "sector": p["sector"],
                     "entry": p["entry"], "limit": p["limit"], "stop": p["stop"],
                     "target": p["target"], "shares": p["shares"], "risk": p["risk"],
                     "status": "pending"})
    if rows:
        j = pd.concat([j, pd.DataFrame(rows)], ignore_index=True)
    return j[JCOLS]


def track_record(j, bars):
    lines = []
    closed = j[j["status"].isin(CLOSED)]
    if len(closed):
        wins = int((closed["R"] > 0).sum())
        lines.append(f"Closed trades: {len(closed)}. Win rate {wins / len(closed):.0%}, "
                     f"average {closed['R'].mean():+.2f}R, total {closed['R'].sum():+.1f}R "
                     f"({money(closed['pnl'].sum())}).")
    else:
        lines.append("Closed trades: none yet.")
    skipped = j["status"].isin(["not filled", "gapped over limit"]).sum()
    if skipped:
        lines.append(f"Orders that never filled: {skipped}.")
    for _, s in j[j["status"] == "open"].iterrows():
        last = float(bars[s["ticker"]]["Close"].iloc[-1]) if s["ticker"] in bars else s["fill"]
        lines.append(f"- Open: {s['ticker']} filled {s['fill_date']} at {money(s['fill'])}, "
                     f"last {money(last)} ({(last - s['fill']) / s['risk']:+.2f}R), "
                     f"stop {money(s['stop'])}, target {money(s['target'])}")
    return lines


# ---------------------------------------------------------------- report

def money(x):
    return f"${x:,.2f}"


def write_report(today, src, total, results, picks, built_fail, no_data, record):
    asof = max((r["asof"] for r in results if "asof" in r), default="n/a")
    counts = {k: sum(1 for r in results if r["checks"].get(k)) for k in
              ["1 near the line", "2 ranges shrinking", "3 volume below avg", "4 no gap-down"]}
    all4 = sum(1 for r in results if r["checks"] and all(r["checks"].values()))

    md = [f"# The Trade Playbook: morning report, {today:%a %b %d, %Y}",
          f"Watchlist: {src.name} ({total} tickers). Prices through the close of {asof}.",
          ""]
    if picks:
        md += ["| # | Ticker | Sector | Entry (buy stop) | Limit | Stop | Take profit | Shares | Risk/sh | R |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for i, p in enumerate(picks, 1):
            md.append(f"| {i} | {p['ticker']} | {p['sector']} | {money(p['entry'])} | {money(p['limit'])} | "
                      f"{money(p['stop'])} | {money(p['target'])} | {p['shares']} | {money(p['risk'])} | {p['R']:.1f} |")
        md.append("")
        for p in picks:
            how = (f"pulled back to the {p['line']}, {p['dist_atr']} ATR above it" if p["route"] == "pullback"
                   else "3-week base, swings narrowing")
            md.append(f"- **{p['ticker']}**{(' (' + p['company'] + ')') if p['company'] else ''}: {how}; "
                      f"ranges {' > '.join('$%.2f' % x for x in p['ranges'])}; risk {p['risk_atr']} ATR; "
                      f"target just below ${p['round_number']:g}; next earnings {p['earnings']}."
                      + (" Already an open trade, not a new entry." if p.get("already_open") else ""))
    else:
        md.append("**No trades today.** Nothing passed all four checks and the building rules.")
    md += ["", f"Funnel: {total} on the list, {len(no_data)} skipped for missing data. Check 1: {counts['1 near the line']}, "
           f"check 2: {counts['2 ranges shrinking']}, check 3: {counts['3 volume below avg']}, "
           f"check 4: {counts['4 no gap-down']}. All four: {all4}. Fully built: {len(picks)} shown."]
    md += ["", "## Track record", ""] + record
    if built_fail:
        md += ["", "Passed all four checks but failed building:"]
        md += [f"- {r['ticker']}: {r['fail']}" for r in built_fail]
    if no_data:
        md += ["", "Skipped, not enough data: " + ", ".join(no_data)]
    md += ["", "Mechanical output of your playbook rules, not investment advice. "
           "Look at each chart before placing an order. Place entries as buy stop-limit orders: "
           "they trigger at the entry price and never pay more than the limit."]
    text = "\n".join(md)

    rows = "".join(
        f"<tr><td>{i}</td><td><b>{p['ticker']}</b><br><small>{html.escape(p['company'])}</small></td>"
        f"<td>{html.escape(p['sector'])}</td><td>{money(p['entry'])}</td><td>{money(p['limit'])}</td><td>{money(p['stop'])}</td>"
        f"<td>{money(p['target'])}</td><td>{p['shares']}</td><td>{money(p['risk'])}</td><td>{p['R']:.1f}</td></tr>"
        for i, p in enumerate(picks, 1))
    notes = "".join(f"<li>{html.escape(l[2:]).replace('**', '')}</li>" for l in md if l.startswith("- **"))
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Trade report {today}</title>
<style>body{{font:14px/1.5 -apple-system,Segoe UI,Arial,sans-serif;color:#1d1d1f;max-width:860px;margin:24px auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:6px 8px;border-bottom:1px solid #ddd;text-align:left}}
th{{background:#f4f4f5}}small{{color:#666}}</style></head><body>
<h2>The Trade Playbook: morning report, {today:%a %b %d, %Y}</h2>
<p>Watchlist: {html.escape(src.name)} ({total} tickers). Prices through the close of {asof}.</p>
{'<table><tr><th>#</th><th>Ticker</th><th>Sector</th><th>Entry (buy stop)</th><th>Limit</th><th>Stop</th><th>Take profit</th><th>Shares</th><th>Risk/sh</th><th>R</th></tr>' + rows + '</table>' if picks else '<p><b>No trades today.</b> Nothing passed all four checks and the building rules.</p>'}
<ul>{notes}</ul>
<h3>Track record</h3>{''.join(f'<p>{html.escape(l)}</p>' for l in record)}
<p><small>{html.escape(md[-1])}</small></p></body></html>"""

    REPORTS.mkdir(exist_ok=True)
    stem = f"{today:%Y-%m-%d}"
    (REPORTS / f"{stem}.md").write_text(text, encoding="utf-8")
    (REPORTS / f"{stem}.html").write_text(page, encoding="utf-8")
    summary = {"date": stem, "asof": asof, "watchlist": src.name, "picks": picks,
               "markdown": str(REPORTS / f"{stem}.md"), "html": str(REPORTS / f"{stem}.html")}
    (REPORTS / "latest.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path)
    args = ap.parse_args()

    src = args.file or newest_watchlist()
    watch = load_watchlist(src)
    tickers = list(watch)
    bars = download(tickers)
    no_data = [t for t in tickers if t not in bars]

    results = [analyze(t, bars[t]) for t in tickers if t in bars]
    built = sorted((r for r in results if r["fail"] is None), key=lambda r: -r["R"])
    built_fail = [r for r in results if r["fail"] and r["checks"] and all(r["checks"].values())]
    no_data += [f"{r['ticker']} ({r['fail']})" for r in results if not r["checks"]]

    survivors = []
    for r in built:
        when, known = next_earnings(r["ticker"])
        # The watchlist's own earnings date counts too; take the earliest upcoming one.
        sheet = watch[r["ticker"]]["earnings"]
        if sheet and sheet >= dt.date.today() and (when is None or sheet < when):
            when, known = sheet, True
        if when and (when - dt.date.today()).days <= EARNINGS_LOOKAHEAD_DAYS:
            r["fail"] = f"earnings on {when} falls inside the hold"
            built_fail.append(r)
            continue
        r["earnings"] = str(when) if when else ("none scheduled" if known else "UNKNOWN, check before trading")
        r["company"] = watch[r["ticker"]]["company"]
        r["sector"] = sector_of(r["ticker"], watch[r["ticker"]]["sector"])
        survivors.append(r)

    picks = diversify(survivors, TOP_N)
    today = dt.date.today()

    journal = load_journal()
    live = journal.loc[journal["status"].isin(["pending", "open"]), "ticker"]
    missing = sorted(set(live) - set(bars))
    if missing:
        bars.update(download(missing))
    journal = settle(journal, bars)
    journal = log_picks(journal, picks, today)
    journal.to_csv(JOURNAL, index=False)
    record = track_record(journal, bars)

    print(write_report(today, src, len(tickers), results, picks, built_fail, no_data, record))


if __name__ == "__main__":
    main()
