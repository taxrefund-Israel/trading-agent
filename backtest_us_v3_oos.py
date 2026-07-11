"""
US Momentum Rotation v3 — OUT-OF-SAMPLE validation: Jul 2016 - Jun 2021.
Identical rules/engine to v1/v2; only the period (and correct MMF yields) change.
"""
from __future__ import annotations
import warnings
import pandas as pd
import backtest_us_v1 as eng
from backtest_us_v1 import fetch_closes, run_variant, UNIVERSE, BENCHMARKS, TAX_RATE

warnings.filterwarnings("ignore")

SIM_START = pd.Timestamp("2016-07-01")
SIM_END   = pd.Timestamp("2021-06-30")
eng.FETCH_START = "2015-01-01"
eng.FETCH_END   = "2021-07-01"
# correct money-market yields for the period
eng.MMF_YIELD.update({2015: 0.001, 2016: 0.003, 2017: 0.009, 2018: 0.018,
                      2019: 0.022, 2020: 0.004, 2021: 0.0005})


def main():
    print("Fetching data (2015-2021)...")
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

    mf = {}
    for d in in_sim:
        mf.setdefault((d.year, d.month), d)
    month_firsts = set(mf.values())

    print(f"\n{'='*100}")
    print(f"  OUT-OF-SAMPLE  |  Jul 2016 - Jun 2021  |  $100,000")
    print(f"{'='*100}")
    bench_net = {}
    for name, tk in BENCHMARKS.items():
        s = bench[tk]
        s = s[(s.index >= SIM_START) & (s.index <= SIM_END)]
        gross = s.iloc[-1] / s.iloc[0] - 1
        net = (gross - max(0, gross) * TAX_RATE) * 100
        bench_net[name] = net
        print(f"  B&H {name:<10} gross {gross*100:>+7.2f}%   net(25% tax) {net:>+7.2f}%   CAGR(net) {((1+net/100)**0.2-1)*100:>+5.2f}%")

    grid = []
    for top_n in (4, 5, 6, 7, 8):
        for buf_mult in (1.5, 2.0, 2.5):
            keep = int(top_n * buf_mult)
            grid.append(dict(top_n=top_n, keep_rank=keep, trend_filter=False,
                             inv_vol=False, abs_mom=False,
                             label=f"Top{top_n} buf{keep}"))

    print(f"\n  {'Variant':<28} {'Return':>9} {'CAGR':>7} {'MaxDD':>8} {'Sells':>6} {'vs SPX':>8} {'vs DJI':>8}")
    print(f"  {'-'*84}")
    beats = 0
    for v in grid:
        r = run_variant(px, spx, days, month_firsts, **v)
        a1 = r["ret"] - bench_net["S&P 500"]; a2 = r["ret"] - bench_net["Dow Jones"]
        if a1 > 0 and a2 > 0:
            beats += 1
            flag = ""
        else:
            flag = "   *LOSES*"
        star = "  <== main pick" if v["label"] == "Top5 buf12" else ""
        print(f"  {r['label']:<28} {r['ret']:>+8.2f}% {r['cagr']:>+6.2f}% {r['max_dd']:>7.1f}% "
              f"{r['sells']:>6} {a1:>+7.2f}% {a2:>+7.2f}%{flag}{star}")

    print(f"\n  {beats}/{len(grid)} variants beat BOTH indexes out-of-sample")


if __name__ == "__main__":
    main()
