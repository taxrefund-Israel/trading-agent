"""
v6b — focused grid over the momentum engine to find a config that beats
ALL THREE indexes (SPX, NDX, DJI) net in BOTH 5yr periods, with positive
alpha in the LIVE window and better MaxDD than the current Top5.

Axes: top_n x sector_cap x momentum lookback (12-1 vs 6-1) x vol_adj.
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
from backtest_us_v6_next import (run_momentum, BENCH, PERIODS, TAX_RATE,
                                 INITIAL_CASH, stats)

eng.FETCH_START = "2015-01-01"
eng.FETCH_END   = "2026-09-06"
eng.MMF_YIELD.update({2015: 0.001, 2016: 0.003, 2017: 0.009, 2018: 0.018,
                      2019: 0.022, 2020: 0.004, 2021: 0.0005})


def main():
    print("Fetching data...")
    closes = fetch_closes(UNIVERSE)
    bench = fetch_closes(list(BENCH.values()))
    print(f"  {len(closes)} stocks fetched")

    spx_full = bench["^GSPC"]
    ndx_full = bench["^NDX"]
    days_full = spx_full.index
    px_full = pd.DataFrame(closes).sort_index().ffill().reindex(days_full).ffill()

    grid = []
    for top_n in (5, 6, 7):
        for cap in (1, 2):
            for lb in (252, 126):
                for va in (False, True):
                    grid.append(dict(top_n=top_n, keep_rank=int(top_n * 2.4),
                                     sector_cap=cap, lb=lb, vol_adj=va))

    # collect per-period results per config
    rows = {}
    for pname, p_start, p_end in PERIODS:
        sim_days = days_full[(days_full >= p_start) & (days_full <= p_end)]
        if len(sim_days) < 5:
            continue
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
        month_firsts, weekly = set(mf.values()), set(wf.values())

        bench_net = {}
        for name, tk in BENCH.items():
            s = bench[tk]
            s = s[(s.index >= p_start) & (s.index <= p_end)]
            gross = s.iloc[-1] / s.iloc[0] - 1
            bench_net[name] = (gross - max(0, gross) * TAX_RATE) * 100

        for cfg in grid:
            key = (cfg["top_n"], cfg["sector_cap"], cfg["lb"], cfg["vol_adj"])
            bk, vals = run_momentum(px, bull, days, month_firsts, weekly, **cfg)
            v, dd = stats(vals[len(days) - len(in_sim):], in_sim)
            ret = (bk.cash / INITIAL_CASH - 1) * 100
            rows.setdefault(key, {})[pname] = {
                "ret": ret, "dd": dd,
                "beats_all": all(ret > b for b in bench_net.values()),
                "a_spx": ret - bench_net["S&P 500"],
            }
        print(f"  done: {pname}")

    P_MAIN, P_OOS, P_LIVE = [p[0] for p in PERIODS]
    print(f"\n  {'Config (topN/cap/lb/volAdj)':<30} {'MAIN ret':>9} {'DD':>7} {'OOS ret':>9} {'DD':>7} {'LIVE':>7}  {'beats all 3?':>14}")
    print(f"  {'-'*96}")
    winners = []
    for key, r in sorted(rows.items(), key=lambda kv: -(kv[1][P_MAIN]["ret"] + kv[1][P_OOS]["ret"])):
        m, o, l = r[P_MAIN], r[P_OOS], r[P_LIVE]
        both = m["beats_all"] and o["beats_all"]
        live_ok = l["a_spx"] > 0
        tag = ("YES" if both else "no ") + (" +live" if live_ok else "  -live")
        label = f"Top{key[0]} cap{key[1]} lb{key[2]} va{int(key[3])}"
        if both and live_ok:
            winners.append((key, r))
        print(f"  {label:<30} {m['ret']:>+8.1f}% {m['dd']:>6.1f}% {o['ret']:>+8.1f}% {o['dd']:>6.1f}% "
              f"{l['ret']:>+6.1f}%  {tag:>14}")

    print(f"\n  {len(winners)}/{len(rows)} configs beat all 3 indexes in BOTH 5yr periods AND have positive live alpha")


if __name__ == "__main__":
    main()
