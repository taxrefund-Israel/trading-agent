"""
US Momentum Rotation v2 — sensitivity / robustness check around v1 winners.
Grid: top_n x keep_rank + regime SMA150 variant + mid-month rebalance shift.
Reuses backtest_us_v1 engine.
"""
from __future__ import annotations
import warnings
import pandas as pd
from backtest_us_v1 import (fetch_closes, run_variant, UNIVERSE, BENCHMARKS,
                            SIM_START, SIM_END, TAX_RATE)

warnings.filterwarnings("ignore")


def main():
    print("Fetching data...")
    closes = fetch_closes(UNIVERSE)
    bench = fetch_closes(list(BENCHMARKS.values()))
    print(f"  {len(closes)} stocks fetched")

    spx_full = bench["^GSPC"]
    px_all = pd.DataFrame(closes).sort_index().ffill()
    days_full = spx_full.index
    px_all = px_all.reindex(days_full).ffill()

    sim_days = days_full[(days_full >= SIM_START) & (days_full <= SIM_END)]
    start_i = list(days_full).index(sim_days[0])
    px = px_all.iloc[max(0, start_i - 260):]
    spx = spx_full.reindex(px.index).ffill()
    days = px.index
    in_sim = [d for d in days if SIM_START <= d <= SIM_END]

    def month_firsts_shift(shift):
        """shift=0: first trading day of month; shift=k: k-th trading day."""
        by_month = {}
        for d in in_sim:
            by_month.setdefault((d.year, d.month), []).append(d)
        out = set()
        for _, ds in by_month.items():
            out.add(ds[min(shift, len(ds) - 1)])
        return out

    mf0 = month_firsts_shift(0)

    bench_net = {}
    for name, tk in BENCHMARKS.items():
        s = bench[tk]
        s = s[(s.index >= SIM_START) & (s.index <= SIM_END)]
        gross = s.iloc[-1] / s.iloc[0] - 1
        bench_net[name] = (gross - max(0, gross) * TAX_RATE) * 100

    print(f"  SPX net {bench_net['S&P 500']:+.2f}%  DJI net {bench_net['Dow Jones']:+.2f}%\n")

    grid = []
    for top_n in (4, 5, 6, 7, 8):
        for buf_mult in (1.5, 2.0, 2.5):
            keep = int(top_n * buf_mult)
            grid.append(dict(top_n=top_n, keep_rank=keep, trend_filter=False,
                             inv_vol=False, abs_mom=False,
                             label=f"Top{top_n} buf{keep}"))

    print(f"  {'Variant':<28} {'Return':>9} {'CAGR':>7} {'MaxDD':>8} {'Sells':>6} {'vs SPX':>8} {'vs DJI':>8}")
    print(f"  {'-'*84}")
    results = []
    for v in grid:
        r = run_variant(px, spx, days, mf0, **v)
        results.append((v, r))
        a1 = r["ret"] - bench_net["S&P 500"]; a2 = r["ret"] - bench_net["Dow Jones"]
        flag = "" if (a1 > 0 and a2 > 0) else "   *LOSES*"
        print(f"  {r['label']:<28} {r['ret']:>+8.2f}% {r['cagr']:>+6.2f}% {r['max_dd']:>7.1f}% "
              f"{r['sells']:>6} {a1:>+7.2f}% {a2:>+7.2f}%{flag}")

    # timing robustness for the top-2 configs: rebalance on 5th / 10th trading day
    print(f"\n  Timing robustness (rebalance day-of-month shift):")
    top2 = sorted(results, key=lambda x: x[1]["ret"], reverse=True)[:2]
    for v, r0 in top2:
        row = [f"{r0['ret']:+.1f}%(d1)"]
        for shift in (4, 9):
            mfs = month_firsts_shift(shift)
            rs = run_variant(px, spx, days, mfs, **v)
            row.append(f"{rs['ret']:+.1f}%(d{shift+1})")
        print(f"    {v['label']:<24} " + "  ".join(row))


if __name__ == "__main__":
    main()
