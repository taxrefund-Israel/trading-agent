"""
v6c — deep validation of the chosen config: Top6, sector-cap 1, 12-1 momentum,
buffer 14, monthly rebalance, weekly hybrid regime (SPX+NDX vs SMA200).

Evidence of significance:
  1. Continuous decade run Jul 2016 -> Sep 2026 (through 3 bear phases).
  2. Calendar-year breakdown vs SPX/NDX/DJI (gross, tax on sells as usual).
  3. Rolling 12-month windows: % of windows beating SPX (gross curves).
  4. LIVE window (2026-07-11 -> now) exact numbers vs benchmarks.
"""
from __future__ import annotations
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import backtest_us_v1 as eng
from backtest_us_v1 import fetch_closes, UNIVERSE
from backtest_us_v6_next import run_momentum, BENCH, TAX_RATE, INITIAL_CASH

eng.FETCH_START = "2015-01-01"
eng.FETCH_END   = "2026-09-06"
eng.MMF_YIELD.update({2015: 0.001, 2016: 0.003, 2017: 0.009, 2018: 0.018,
                      2019: 0.022, 2020: 0.004, 2021: 0.0005})

CFG = dict(top_n=6, keep_rank=14, sector_cap=1, lb=252, vol_adj=False)
DECADE_START = pd.Timestamp("2016-07-01")
LIVE_START   = pd.Timestamp("2026-07-11")


def run_span(px_full, spx_full, ndx_full, days_full, p_start, p_end, cash0=INITIAL_CASH):
    sim_days = days_full[(days_full >= p_start) & (days_full <= p_end)]
    i0 = list(days_full).index(sim_days[0])
    px = px_full.iloc[max(0, i0 - 280):list(days_full).index(sim_days[-1]) + 1]
    spx = spx_full.reindex(px.index).ffill()
    ndx = ndx_full.reindex(px.index).ffill()
    days = px.index
    in_sim = [d for d in days if p_start <= d <= p_end]
    bull = (spx > spx.rolling(200).mean()) & (ndx > ndx.rolling(200).mean())
    mf, wf = {}, {}
    for d in in_sim:
        mf.setdefault((d.year, d.month), d)
        wf.setdefault((d.isocalendar().year, d.isocalendar().week), d)
    bk, vals = run_momentum(px, bull, days, set(mf.values()), set(wf.values()),
                            cash0=cash0, **CFG)
    curve = pd.Series(vals[len(days) - len(in_sim):], index=in_sim)
    return bk, curve


def main():
    print("Fetching data...")
    closes = fetch_closes(UNIVERSE)
    bench = fetch_closes(list(BENCH.values()))
    print(f"  {len(closes)} stocks fetched\n")

    spx_full = bench["^GSPC"]
    ndx_full = bench["^NDX"]
    dji_full = bench["^DJI"]
    days_full = spx_full.index
    px_full = pd.DataFrame(closes).sort_index().ffill().reindex(days_full).ffill()
    last_day = days_full[-1]

    # ── 1. continuous decade ──────────────────────────────────────────────────
    bk, curve = run_span(px_full, spx_full, ndx_full, days_full, DECADE_START, last_day)
    yrs = len(curve) / 252
    ret = (bk.cash / INITIAL_CASH - 1) * 100
    cagr = ((bk.cash / INITIAL_CASH) ** (1 / yrs) - 1) * 100
    dd = ((curve - curve.cummax()) / curve.cummax()).min() * 100

    def bench_ret(s, a, b, net=True):
        seg = s[(s.index >= a) & (s.index <= b)]
        g = seg.iloc[-1] / seg.iloc[0] - 1
        return (g - max(0, g) * TAX_RATE) * 100 if net else g * 100

    print("=" * 100)
    print(f"  1. CONTINUOUS DECADE  {DECADE_START:%Y-%m-%d} -> {last_day:%Y-%m-%d}  ({yrs:.1f} yrs)")
    print("=" * 100)
    print(f"  Strategy (net, liq-taxed): {ret:+10.1f}%   CAGR {cagr:+.2f}%   MaxDD {dd:.1f}%   sells {bk.sells}   tax ${bk.tax:,.0f}")
    for name, s in (("S&P 500", spx_full), ("NASDAQ-100", ndx_full), ("Dow Jones", dji_full)):
        print(f"  B&H {name:<11} (net):   {bench_ret(s, DECADE_START, last_day):+10.1f}%")

    # ── 2. calendar years ─────────────────────────────────────────────────────
    print(f"\n  2. CALENDAR YEARS (strategy curve vs indexes, gross-of-final-tax)")
    print(f"  {'Year':<6} {'Strategy':>10} {'SPX':>9} {'NDX':>9} {'DJI':>9}  {'beats SPX':>10}")
    print(f"  {'-'*60}")
    wins = 0
    years = sorted(set(curve.index.year))
    for y in years:
        seg = curve[curve.index.year == y]
        if len(seg) < 20:
            continue
        sret = (seg.iloc[-1] / seg.iloc[0] - 1) * 100
        row = [sret]
        for s in (spx_full, ndx_full, dji_full):
            b = s[(s.index >= seg.index[0]) & (s.index <= seg.index[-1])]
            row.append((b.iloc[-1] / b.iloc[0] - 1) * 100)
        beat = sret > row[1]
        wins += beat
        print(f"  {y:<6} {row[0]:>+9.1f}% {row[1]:>+8.1f}% {row[2]:>+8.1f}% {row[3]:>+8.1f}%  {'YES' if beat else 'no':>10}")
    print(f"  -> beats SPX in {wins}/{len(years)} calendar years")

    # ── 3. rolling 12m windows ────────────────────────────────────────────────
    spx_c = spx_full.reindex(curve.index).ffill()
    roll_s = curve.pct_change(252).dropna()
    roll_b = spx_c.pct_change(252).dropna()
    common = roll_s.index.intersection(roll_b.index)
    beat = (roll_s[common] > roll_b[common])
    print(f"\n  3. ROLLING 12-MONTH WINDOWS (daily-stepped, {len(common)} windows)")
    print(f"     strategy beats SPX in {beat.mean()*100:.1f}% of all 1-year windows")
    print(f"     median 12m: strategy {roll_s[common].median()*100:+.1f}%  vs SPX {roll_b[common].median()*100:+.1f}%")

    # ── 4. LIVE window ────────────────────────────────────────────────────────
    bk_l, curve_l = run_span(px_full, spx_full, ndx_full, days_full, LIVE_START, last_day)
    ret_l = (bk_l.cash / INITIAL_CASH - 1) * 100
    dd_l = ((curve_l - curve_l.cummax()) / curve_l.cummax()).min() * 100
    print(f"\n  4. LIVE WINDOW  {LIVE_START:%Y-%m-%d} -> {last_day:%Y-%m-%d}")
    print(f"  Strategy (net):  {ret_l:+7.2f}%   MaxDD {dd_l:.1f}%   sells {bk_l.sells}")
    for name, s in (("S&P 500", spx_full), ("NASDAQ-100", ndx_full), ("Dow Jones", dji_full)):
        print(f"  B&H {name:<11} (net):  {bench_ret(s, LIVE_START, last_day):+7.2f}%")
    print(f"  (current live Top5 strategy actual result over this window: about -15%)")


if __name__ == "__main__":
    main()
