"""
US Momentum Rotation v5b — hybrid regime: bear if EITHER SPX or NDX < its SMA200.
(Re-enter only when BOTH are above.) Weekly check, vs pure SPX / pure NDX.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import backtest_us_v1 as eng
from backtest_us_v1 import fetch_closes, UNIVERSE, TAX_RATE, COMMISSION, INITIAL_CASH, mmf_daily

warnings.filterwarnings("ignore")

eng.FETCH_START = "2015-01-01"
eng.FETCH_END   = "2026-07-01"
eng.MMF_YIELD.update({2015: 0.001, 2016: 0.003, 2017: 0.009, 2018: 0.018,
                      2019: 0.022, 2020: 0.004, 2021: 0.0005})

BENCH = {"S&P 500": "^GSPC", "NASDAQ-100": "^NDX", "Dow Jones": "^DJI"}
PERIODS = [
    ("OOS  2016-2021", pd.Timestamp("2016-07-01"), pd.Timestamp("2021-06-30")),
    ("MAIN 2021-2026", pd.Timestamp("2021-07-01"), pd.Timestamp("2026-06-30")),
]


def run_bull_series(px, bull_series, days, month_firsts, regime_days, *,
                    top_n, keep_rank, label):
    """כמו run_variant2 של v4, אבל מקבל סדרת bull בוליאנית מוכנה."""
    cash = INITIAL_CASH
    pos = {}
    tax_paid = 0.0
    n_sells = 0
    values = []
    bull_state = True

    def sell(sym, p):
        nonlocal cash, tax_paid, n_sells
        h = pos.pop(sym)
        proceeds = h["qty"] * p
        sc = proceeds * COMMISSION
        gain = proceeds - h["cost"] - h["comm"] - sc
        tax = max(0.0, gain * TAX_RATE)
        cash += proceeds - sc - tax
        tax_paid += tax
        n_sells += 1

    def buy_slots(i, prices):
        nonlocal cash
        slots = top_n - len(pos)
        if slots <= 0 or i < 252:
            return
        mom = (px.iloc[i - 21] / px.iloc[i - 252] - 1).dropna()
        ranked = mom.sort_values(ascending=False)
        pool = [s for s in ranked.index if s not in pos and not np.isnan(prices[s])]
        per_slot = cash * 0.98 / slots
        filled = 0
        for s in pool:
            if filled >= slots:
                break
            p = prices[s]
            qty = int(per_slot / (p * (1 + COMMISSION)))
            if qty < 1:
                continue
            bc = qty * p * COMMISSION
            cash -= qty * p + bc
            pos[s] = {"qty": qty, "cost": qty * p, "comm": bc}
            filled += 1

    for i, day in enumerate(days):
        cash *= (1 + mmf_daily(day))
        prices = px.iloc[i]

        if day in regime_days and i >= 252:
            prev = bull_state
            bull_state = bool(bull_series.iloc[i])
            if prev and not bull_state:
                for sym in list(pos.keys()):
                    p = prices[sym]
                    if not np.isnan(p):
                        sell(sym, p)
            elif not prev and bull_state and day not in month_firsts:
                buy_slots(i, prices)

        if day in month_firsts and i >= 252 and bull_state:
            mom = (px.iloc[i - 21] / px.iloc[i - 252] - 1).dropna()
            ranked = mom.sort_values(ascending=False)
            rank_of = {s: r + 1 for r, s in enumerate(ranked.index)}
            for sym in list(pos.keys()):
                r = rank_of.get(sym)
                if r is not None and r <= keep_rank:
                    continue
                p = prices[sym]
                if not np.isnan(p):
                    sell(sym, p)
            buy_slots(i, prices)

        held = sum(h["qty"] * prices[s] for s, h in pos.items() if not np.isnan(prices[s]))
        values.append(cash + held)

    last = px.iloc[-1]
    for sym in list(pos.keys()):
        p = last[sym]
        if not np.isnan(p):
            sell(sym, p)

    vals = pd.Series(values, index=days)
    dd = ((vals - vals.cummax()) / vals.cummax()).min() * 100
    ret = (cash / INITIAL_CASH - 1) * 100
    return {"label": label, "ret": ret, "cagr": ((cash / INITIAL_CASH) ** 0.2 - 1) * 100,
            "max_dd": dd, "sells": n_sells}


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

        spx_bull = spx > spx.rolling(200).mean()
        ndx_bull = ndx > ndx.rolling(200).mean()
        series = {
            "SPX only":          spx_bull,
            "NDX only":          ndx_bull,
            "Hybrid (both up)":  spx_bull & ndx_bull,
        }

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

        print(f"\n{'='*104}")
        print(f"  {pname}  |  SPX {bench_net['S&P 500']:+.1f}%  NDX {bench_net['NASDAQ-100']:+.1f}%  (net)")
        print(f"{'='*104}")
        print(f"  {'Variant':<44} {'Return':>9} {'CAGR':>7} {'MaxDD':>8} {'Sells':>6} {'vs NDX':>8}")
        print(f"  {'-'*88}")
        for top_n, keep in ((5, 12), (6, 15)):
            for rname, bs in series.items():
                r = run_bull_series(px, bs, days, month_firsts, weekly,
                                    top_n=top_n, keep_rank=keep,
                                    label=f"Top{top_n} buf{keep} | {rname}")
                print(f"  {r['label']:<44} {r['ret']:>+8.2f}% {r['cagr']:>+6.2f}% "
                      f"{r['max_dd']:>7.1f}% {r['sells']:>6} "
                      f"{r['ret'] - bench_net['NASDAQ-100']:>+7.2f}%")
            print(f"  {'-'*88}")


if __name__ == "__main__":
    main()
