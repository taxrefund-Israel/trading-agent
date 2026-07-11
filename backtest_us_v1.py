"""
US Momentum Rotation Backtest v1 — beat S&P 500 & Dow Jones
Period: July 2021 - June 2026 (5 years)

Lessons from TA-125 v9-13 + NASDAQ failure:
  * Daily TA trading in US mega-cap indexes fails: tax drag (237 sells) +
    missing the NVDA-style mega-winners that drive cap-weighted returns.
  * Fix: LOW-turnover monthly momentum rotation (12-1) with a rank BUFFER
    (only sell when a holding falls far in rank -> winners like NVDA are
    held for years), plus a market-regime filter (SPX vs SMA200) that
    moves to money-market during bear phases (2022).

Accounting (same conventions as TA-125 backtests):
  * 25% tax on realized gains per sale (no loss offset - conservative)
  * commission 0.05% per side
  * FINAL LIQUIDATION IS TAXED for the strategy, benchmarks taxed on
    final gain too -> apples-to-apples net comparison.
"""
from __future__ import annotations
import warnings
import yfinance as yf
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

INITIAL_CASH = 100_000.0
COMMISSION   = 0.0005
TAX_RATE     = 0.25

SIM_START  = pd.Timestamp("2021-07-01")
SIM_END    = pd.Timestamp("2026-06-30")
FETCH_START = "2020-01-01"
FETCH_END   = "2026-07-01"

# US money-market approx annual yields
MMF_YIELD = {2020: 0.003, 2021: 0.0005, 2022: 0.015, 2023: 0.050,
             2024: 0.052, 2025: 0.043, 2026: 0.040}

# ~60 mega/large caps, all listed well before 2020, sector-diverse.
# (Survivorship caveat: fixed current universe; mitigated by using only
#  names that were already S&P100-scale in 2020.)
UNIVERSE = [
    # Tech / semis / software
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","ORCL","CRM",
    "ADBE","AMD","QCOM","TXN","INTC","CSCO","IBM","NOW","INTU","MU",
    "AMAT","LRCX","KLAC","NFLX",
    # Financials
    "JPM","BAC","WFC","GS","MS","V","MA","AXP","BRK-B","BLK",
    # Health
    "UNH","JNJ","LLY","ABBV","MRK","PFE","TMO","ABT",
    # Energy
    "XOM","CVX","COP","SLB",
    # Consumer
    "WMT","COST","PG","KO","PEP","MCD","HD","LOW","NKE","DIS",
    # Industrials / other
    "CAT","DE","HON","GE","LMT","RTX","UNP","UPS","LIN","NEE",
    "T","VZ","CMCSA","BA",
]

BENCHMARKS = {"S&P 500": "^GSPC", "Dow Jones": "^DJI"}


def fetch_closes(tickers):
    out = {}
    for t in tickers:
        try:
            df = yf.Ticker(t).history(start=FETCH_START, end=FETCH_END)
            if df.empty or len(df) < 200:
                continue
            s = df["Close"]
            s.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in s.index])
            out[t] = s
        except Exception:
            pass
    return out


def mmf_daily(day):
    return MMF_YIELD.get(day.year, 0.04) / 252


def run_variant(px, spx, days, month_firsts, *, top_n, keep_rank,
                trend_filter, inv_vol, abs_mom, label):
    """px: DataFrame days x tickers (ffilled). spx: Series aligned to days."""
    cash = INITIAL_CASH
    pos = {}          # sym -> dict(qty, cost_basis_total, comm)
    tax_paid = 0.0
    n_sells = n_buys = 0
    values = []
    spx_sma200 = spx.rolling(200).mean()

    tickers = list(px.columns)
    rets = px.pct_change()

    for i, day in enumerate(days):
        # accrue money-market on cash
        cash *= (1 + mmf_daily(day))
        prices = px.iloc[i]

        if day in month_firsts and i >= 252:
            bull = spx.iloc[i] > spx_sma200.iloc[i]

            if not bull:
                # regime exit: liquidate everything
                for sym in list(pos.keys()):
                    p = prices[sym]
                    if np.isnan(p):
                        continue
                    h = pos.pop(sym)
                    proceeds = h["qty"] * p
                    sc = proceeds * COMMISSION
                    gain = proceeds - h["cost"] - h["comm"] - sc
                    tax = max(0.0, gain * TAX_RATE)
                    cash += proceeds - sc - tax
                    tax_paid += tax
                    n_sells += 1
            else:
                mom = px.iloc[i - 21] / px.iloc[i - 252] - 1
                mom = mom.dropna()
                if abs_mom:
                    mom = mom[mom > 0]
                if trend_filter:
                    sma200 = px.iloc[max(0, i - 200):i].mean()
                    mom = mom[prices[mom.index] > sma200[mom.index]]
                ranked = mom.sort_values(ascending=False)
                rank_of = {s: r + 1 for r, s in enumerate(ranked.index)}

                # SELL: holdings that fell below keep_rank (or lost eligibility)
                for sym in list(pos.keys()):
                    r = rank_of.get(sym)
                    if r is not None and r <= keep_rank:
                        continue
                    p = prices[sym]
                    if np.isnan(p):
                        continue
                    h = pos.pop(sym)
                    proceeds = h["qty"] * p
                    sc = proceeds * COMMISSION
                    gain = proceeds - h["cost"] - h["comm"] - sc
                    tax = max(0.0, gain * TAX_RATE)
                    cash += proceeds - sc - tax
                    tax_paid += tax
                    n_sells += 1

                # BUY: fill open slots with best-ranked new names
                slots = top_n - len(pos)
                if slots > 0:
                    cands = [s for s in ranked.index
                             if s not in pos and not np.isnan(prices[s])][:slots]
                    if cands:
                        if inv_vol:
                            vol = rets.iloc[max(0, i - 63):i][cands].std()
                            iv = 1.0 / vol.replace(0, np.nan)
                            w = (iv / iv.sum()).fillna(1.0 / len(cands))
                        else:
                            w = pd.Series(1.0 / len(cands), index=cands)
                        budget = cash * 0.98
                        for s in cands:
                            alloc = budget * w[s]
                            p = prices[s]
                            qty = int(alloc / (p * (1 + COMMISSION)))
                            if qty < 1:
                                continue
                            bc = qty * p * COMMISSION
                            cash -= qty * p + bc
                            pos[s] = {"qty": qty, "cost": qty * p, "comm": bc}
                            n_buys += 1

        held_val = sum(h["qty"] * prices[s] for s, h in pos.items()
                       if not np.isnan(prices[s]))
        values.append(cash + held_val)

    # final liquidation, taxed
    last = px.iloc[-1]
    for sym, h in pos.items():
        p = last[sym]
        if np.isnan(p):
            continue
        proceeds = h["qty"] * p
        sc = proceeds * COMMISSION
        gain = proceeds - h["cost"] - h["comm"] - sc
        tax = max(0.0, gain * TAX_RATE)
        cash += proceeds - sc - tax
        tax_paid += tax

    vals = pd.Series(values, index=days)
    peak = vals.cummax()
    max_dd = ((vals - peak) / peak).min() * 100

    final = cash
    ret = (final / INITIAL_CASH - 1) * 100
    cagr = ((final / INITIAL_CASH) ** (1 / 5) - 1) * 100
    return {"label": label, "final": final, "ret": ret, "cagr": cagr,
            "max_dd": max_dd, "tax": tax_paid, "sells": n_sells, "buys": n_buys}


def main():
    print("Fetching data...")
    closes = fetch_closes(UNIVERSE)
    bench = fetch_closes(list(BENCHMARKS.values()))
    print(f"  {len(closes)} stocks fetched")

    spx_full = bench["^GSPC"]
    px_all = pd.DataFrame(closes).sort_index().ffill()
    # align everything to SPX trading days
    days_full = spx_full.index
    px_all = px_all.reindex(days_full).ffill()

    sim_mask = (days_full >= SIM_START) & (days_full <= SIM_END)
    sim_days = days_full[sim_mask]
    # keep 252 lookback available: run loop over full index but rebalance in sim
    # -> simpler: slice px so iloc positions align, with lookback inside slice
    start_i = list(days_full).index(sim_days[0])
    lb_i = max(0, start_i - 260)
    px = px_all.iloc[lb_i:]
    spx = spx_full.reindex(px.index).ffill()
    days = px.index
    in_sim = [d for d in days if SIM_START <= d <= SIM_END]

    mf = {}
    for d in in_sim:
        key = (d.year, d.month)
        if key not in mf:
            mf[key] = d
    month_firsts = set(mf.values())

    # benchmarks net of 25% tax on final gain
    print(f"\n{'='*100}")
    print(f"  US MOMENTUM ROTATION v1  |  Jul 2021 - Jun 2026  |  $100,000")
    print(f"{'='*100}")
    bench_net = {}
    for name, tk in BENCHMARKS.items():
        s = bench[tk]
        s = s[(s.index >= SIM_START) & (s.index <= SIM_END)]
        gross = s.iloc[-1] / s.iloc[0] - 1
        net = (gross - max(0, gross) * TAX_RATE) * 100
        bench_net[name] = net
        print(f"  B&H {name:<10} gross {gross*100:>+7.2f}%   net(25% tax) {net:>+7.2f}%   CAGR(net) {((1+net/100)**0.2-1)*100:>+5.2f}%")

    variants = [
        dict(top_n=10, keep_rank=15, trend_filter=False, inv_vol=False, abs_mom=False, label="A  Top10 EW, buffer15"),
        dict(top_n=10, keep_rank=25, trend_filter=False, inv_vol=False, abs_mom=False, label="B  Top10 EW, buffer25 (lazy)"),
        dict(top_n=8,  keep_rank=16, trend_filter=False, inv_vol=False, abs_mom=False, label="C  Top8  EW, buffer16"),
        dict(top_n=5,  keep_rank=12, trend_filter=False, inv_vol=False, abs_mom=False, label="D  Top5  EW, buffer12 (concentrated)"),
        dict(top_n=10, keep_rank=15, trend_filter=False, inv_vol=True,  abs_mom=False, label="E  Top10 InvVol, buffer15"),
        dict(top_n=10, keep_rank=15, trend_filter=True,  inv_vol=False, abs_mom=False, label="F  Top10 EW, buffer15 + own SMA200"),
        dict(top_n=10, keep_rank=15, trend_filter=False, inv_vol=False, abs_mom=True,  label="G  Top10 EW, buffer15 + abs mom>0"),
        dict(top_n=8,  keep_rank=20, trend_filter=True,  inv_vol=False, abs_mom=True,  label="H  Top8 EW buf20 +SMA200 +abs"),
    ]

    print(f"\n  {'Variant':<40} {'Return':>9} {'CAGR':>7} {'MaxDD':>8} {'Tax $':>10} {'Sells':>6} {'vs SPX':>8} {'vs DJI':>8}")
    print(f"  {'-'*100}")
    results = []
    for v in variants:
        r = run_variant(px, spx, days, month_firsts, **v)
        results.append(r)
        a_spx = r["ret"] - bench_net["S&P 500"]
        a_dji = r["ret"] - bench_net["Dow Jones"]
        flag = "  <== BEATS BOTH" if a_spx > 0 and a_dji > 0 else ""
        print(f"  {r['label']:<40} {r['ret']:>+8.2f}% {r['cagr']:>+6.2f}% {r['max_dd']:>7.1f}% "
              f"{r['tax']:>10,.0f} {r['sells']:>6} {a_spx:>+7.2f}% {a_dji:>+7.2f}%{flag}")

    best = max(results, key=lambda r: r["ret"])
    print(f"\n  BEST: {best['label']}  ->  ${best['final']:,.0f}  ({best['ret']:+.2f}%, CAGR {best['cagr']:+.2f}%)")


if __name__ == "__main__":
    main()
