"""
US Momentum Rotation v4 — regime-check frequency test.
Momentum rebalance stays MONTHLY. Regime filter (SPX vs SMA200) is checked:
  M = monthly (baseline, as v1-v3)
  W = weekly  (first trading day of each week)
  D = daily
  D2 = daily with 2% hysteresis band (exit below SMA200*0.98, re-enter above SMA200*1.02)
On bear->bull recovery mid-month, buys top-N immediately (no waiting for month start).
Runs both periods: Jul 2016-Jun 2021 (OOS) and Jul 2021-Jun 2026.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import backtest_us_v1 as eng
from backtest_us_v1 import fetch_closes, UNIVERSE, BENCHMARKS, TAX_RATE, COMMISSION, INITIAL_CASH, mmf_daily

warnings.filterwarnings("ignore")

eng.FETCH_START = "2015-01-01"
eng.FETCH_END   = "2026-07-01"
eng.MMF_YIELD.update({2015: 0.001, 2016: 0.003, 2017: 0.009, 2018: 0.018,
                      2019: 0.022, 2020: 0.004, 2021: 0.0005})

PERIODS = [
    ("OOS  2016-2021", pd.Timestamp("2016-07-01"), pd.Timestamp("2021-06-30")),
    ("MAIN 2021-2026", pd.Timestamp("2021-07-01"), pd.Timestamp("2026-06-30")),
]


def run_variant2(px, spx, days, month_firsts, regime_days, *, top_n, keep_rank,
                 band, label):
    cash = INITIAL_CASH
    pos = {}
    tax_paid = 0.0
    n_sells = n_buys = 0
    values = []
    spx_sma200 = spx.rolling(200).mean()
    rets = px.pct_change()
    bull_state = True  # resolved at first regime check

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
        nonlocal cash, n_buys
        slots = top_n - len(pos)
        if slots <= 0 or i < 252:
            return
        mom = (px.iloc[i - 21] / px.iloc[i - 252] - 1).dropna()
        ranked = mom.sort_values(ascending=False)
        cands = [s for s in ranked.index
                 if s not in pos and not np.isnan(prices[s])][:slots]
        if not cands:
            return
        budget = cash * 0.98
        w = 1.0 / len(cands)
        for s in cands:
            p = prices[s]
            qty = int(budget * w / (p * (1 + COMMISSION)))
            if qty < 1:
                continue
            bc = qty * p * COMMISSION
            cash -= qty * p + bc
            pos[s] = {"qty": qty, "cost": qty * p, "comm": bc}
            n_buys += 1

    for i, day in enumerate(days):
        cash *= (1 + mmf_daily(day))
        prices = px.iloc[i]

        # ── regime check ──
        if day in regime_days and i >= 252:
            sma = spx_sma200.iloc[i]
            c = spx.iloc[i]
            prev = bull_state
            if bull_state and c < sma * (1 - band):
                bull_state = False
            elif not bull_state and c > sma * (1 + band):
                bull_state = True
            if prev and not bull_state:
                for sym in list(pos.keys()):
                    p = prices[sym]
                    if not np.isnan(p):
                        sell(sym, p)
            elif not prev and bull_state and day not in month_firsts:
                buy_slots(i, prices)  # immediate re-entry mid-month

        # ── monthly momentum rebalance ──
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

        held_val = sum(h["qty"] * prices[s] for s, h in pos.items()
                       if not np.isnan(prices[s]))
        values.append(cash + held_val)

    last = px.iloc[-1]
    for sym in list(pos.keys()):
        p = last[sym]
        if not np.isnan(p):
            sell(sym, p)

    vals = pd.Series(values, index=days)
    dd = ((vals - vals.cummax()) / vals.cummax()).min() * 100
    ret = (cash / INITIAL_CASH - 1) * 100
    return {"label": label, "ret": ret,
            "cagr": ((cash / INITIAL_CASH) ** 0.2 - 1) * 100,
            "max_dd": dd, "tax": tax_paid, "sells": n_sells}


def main():
    print("Fetching data (2015-2026)...")
    closes = fetch_closes(UNIVERSE)
    bench = fetch_closes(list(BENCHMARKS.values()))
    print(f"  {len(closes)} stocks fetched")

    spx_full = bench["^GSPC"]
    days_full = spx_full.index
    px_full = pd.DataFrame(closes).sort_index().ffill().reindex(days_full).ffill()

    for pname, p_start, p_end in PERIODS:
        sim_days = days_full[(days_full >= p_start) & (days_full <= p_end)]
        start_i = list(days_full).index(sim_days[0])
        px = px_full.iloc[max(0, start_i - 260):list(days_full).index(sim_days[-1]) + 1]
        spx = spx_full.reindex(px.index).ffill()
        days = px.index
        in_sim = [d for d in days if p_start <= d <= p_end]

        mf, wf = {}, {}
        for d in in_sim:
            mf.setdefault((d.year, d.month), d)
            wf.setdefault((d.isocalendar().year, d.isocalendar().week), d)
        month_firsts = set(mf.values())
        weekly = set(wf.values())
        daily = set(in_sim)

        bench_net = {}
        for name, tk in BENCHMARKS.items():
            s = bench[tk]
            s = s[(s.index >= p_start) & (s.index <= p_end)]
            gross = s.iloc[-1] / s.iloc[0] - 1
            bench_net[name] = (gross - max(0, gross) * TAX_RATE) * 100

        print(f"\n{'='*104}")
        print(f"  {pname}  |  SPX net {bench_net['S&P 500']:+.2f}%  DJI net {bench_net['Dow Jones']:+.2f}%")
        print(f"{'='*104}")
        print(f"  {'Variant':<38} {'Return':>9} {'CAGR':>7} {'MaxDD':>8} {'Tax $':>9} {'Sells':>6} {'vs SPX':>8}")
        print(f"  {'-'*94}")

        for top_n, keep in ((5, 12), (6, 15)):
            for mode, rdays, band in (("M  monthly", month_firsts, 0.0),
                                      ("W  weekly",  weekly,       0.0),
                                      ("D  daily",   daily,        0.0),
                                      ("D2 daily+2%band", daily,   0.02)):
                r = run_variant2(px, spx, days, month_firsts, rdays,
                                 top_n=top_n, keep_rank=keep, band=band,
                                 label=f"Top{top_n} buf{keep} | {mode}")
                a = r["ret"] - bench_net["S&P 500"]
                print(f"  {r['label']:<38} {r['ret']:>+8.2f}% {r['cagr']:>+6.2f}% "
                      f"{r['max_dd']:>7.1f}% {r['tax']:>9,.0f} {r['sells']:>6} {a:>+7.2f}%")
            print(f"  {'-'*94}")


if __name__ == "__main__":
    main()
