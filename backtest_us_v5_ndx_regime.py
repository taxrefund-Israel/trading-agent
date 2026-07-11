"""
US Momentum Rotation v5 — regime-index comparison: SPX vs NDX (NASDAQ-100).
Weekly regime check (the chosen mode), Top5 buf12 & Top6 buf15, both periods.
Everything else identical to v4.
"""
from __future__ import annotations
import warnings
import pandas as pd
import backtest_us_v1 as eng
from backtest_us_v1 import fetch_closes, UNIVERSE, TAX_RATE
from backtest_us_v4_regime import run_variant2

warnings.filterwarnings("ignore")

eng.FETCH_START = "2015-01-01"
eng.FETCH_END   = "2026-07-01"
eng.MMF_YIELD.update({2015: 0.001, 2016: 0.003, 2017: 0.009, 2018: 0.018,
                      2019: 0.022, 2020: 0.004, 2021: 0.0005})

BENCH = {"S&P 500": "^GSPC", "Dow Jones": "^DJI", "NASDAQ-100": "^NDX"}
PERIODS = [
    ("OOS  2016-2021", pd.Timestamp("2016-07-01"), pd.Timestamp("2021-06-30")),
    ("MAIN 2021-2026", pd.Timestamp("2021-07-01"), pd.Timestamp("2026-06-30")),
]


def main():
    print("Fetching data (2015-2026)...")
    closes = fetch_closes(UNIVERSE)
    bench = fetch_closes(list(BENCH.values()))
    print(f"  {len(closes)} stocks fetched")

    spx_full = bench["^GSPC"]
    ndx_full = bench["^NDX"]
    days_full = spx_full.index
    px_full = pd.DataFrame(closes).sort_index().ffill().reindex(days_full).ffill()

    for pname, p_start, p_end in PERIODS:
        sim_days = days_full[(days_full >= p_start) & (days_full <= p_end)]
        start_i = list(days_full).index(sim_days[0])
        px = px_full.iloc[max(0, start_i - 260):list(days_full).index(sim_days[-1]) + 1]
        spx = spx_full.reindex(px.index).ffill()
        ndx = ndx_full.reindex(px.index).ffill()
        days = px.index
        in_sim = [d for d in days if p_start <= d <= p_end]

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

        print(f"\n{'='*106}")
        print(f"  {pname}  |  SPX {bench_net['S&P 500']:+.1f}%  DJI {bench_net['Dow Jones']:+.1f}%  NDX {bench_net['NASDAQ-100']:+.1f}%  (net)")
        print(f"{'='*106}")
        print(f"  {'Variant':<40} {'Return':>9} {'CAGR':>7} {'MaxDD':>8} {'Sells':>6} {'vs SPX':>8} {'vs NDX':>8}")
        print(f"  {'-'*96}")

        for top_n, keep in ((5, 12), (6, 15)):
            for rname, rseries in (("SPX regime", spx), ("NDX regime", ndx)):
                r = run_variant2(px, rseries, days, month_firsts, weekly,
                                 top_n=top_n, keep_rank=keep, band=0.0,
                                 label=f"Top{top_n} buf{keep} | weekly {rname}")
                a1 = r["ret"] - bench_net["S&P 500"]
                a3 = r["ret"] - bench_net["NASDAQ-100"]
                print(f"  {r['label']:<40} {r['ret']:>+8.2f}% {r['cagr']:>+6.2f}% "
                      f"{r['max_dd']:>7.1f}% {r['sells']:>6} {a1:>+7.2f}% {a3:>+7.2f}%")
            print(f"  {'-'*96}")


if __name__ == "__main__":
    main()
