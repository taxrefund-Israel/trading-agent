"""
US Strategy Search v6 — after live underperformance (Jul-Aug 2026: -15% vs SPX +1.3%).

Diagnosis of failure: Top5 momentum loaded 100% into one theme (semis) -> sector
crash hit all 5 at once. Index barely moved.

Candidate fixes tested here (long-only, no leverage, no options):
  CUR    current strategy (Top5 buf12, hybrid weekly regime)      - baseline
  A1     Top10 buf20, equal-weight, SECTOR CAP 2                  - diversification
  A2     Top10 buf20, vol-adjusted momentum score, sector cap 2   - risk-adjusted ranking
  A3     Top8  buf16, vol-adjusted, sector cap 2
  MR     daily RSI(2) mean-reversion dip-buying (bull regime, own SMA200)
  BL     blend: 60% A2 + 40% MR (separate pools)

Periods: 5yr main (Sep 2021 - Aug 2026), OOS (Jul 2016 - Jun 2021),
LIVE window (2026-07-11 -> today) where the current strategy failed.
Accounting: 25% tax per realized gain (no loss offset), 0.05% commission,
final liquidation taxed. Benchmarks net of 25% on final gain.
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

eng.FETCH_START = "2015-01-01"
eng.FETCH_END   = "2026-09-06"
eng.MMF_YIELD.update({2015: 0.001, 2016: 0.003, 2017: 0.009, 2018: 0.018,
                      2019: 0.022, 2020: 0.004, 2021: 0.0005})

COMMISSION   = 0.0005
TAX_RATE     = 0.25
INITIAL_CASH = 100_000.0
mmf_daily    = eng.mmf_daily

SECTOR = {}
for s in ["NVDA","AVGO","AMD","QCOM","TXN","INTC","MU","AMAT","LRCX","KLAC"]:
    SECTOR[s] = "semis"
for s in ["AAPL","MSFT","GOOGL","META","ORCL","CRM","ADBE","CSCO","IBM","NOW","INTU","NFLX"]:
    SECTOR[s] = "tech"
for s in ["JPM","BAC","WFC","GS","MS","V","MA","AXP","BRK-B","BLK"]:
    SECTOR[s] = "fin"
for s in ["UNH","JNJ","LLY","ABBV","MRK","PFE","TMO","ABT"]:
    SECTOR[s] = "health"
for s in ["XOM","CVX","COP","SLB"]:
    SECTOR[s] = "energy"
for s in ["WMT","COST","PG","KO","PEP","MCD","HD","LOW","NKE","DIS","AMZN","TSLA"]:
    SECTOR[s] = "consumer"
for s in ["CAT","DE","HON","GE","LMT","RTX","UNP","UPS","BA"]:
    SECTOR[s] = "industrial"
for s in ["LIN","NEE","T","VZ","CMCSA"]:
    SECTOR[s] = "other"

BENCH = {"S&P 500": "^GSPC", "NASDAQ-100": "^NDX", "Dow Jones": "^DJI"}

PERIODS = [
    ("MAIN 5yr  2021-09 -> 2026-08", pd.Timestamp("2021-09-01"), pd.Timestamp("2026-08-31")),
    ("OOS  5yr  2016-07 -> 2021-06", pd.Timestamp("2016-07-01"), pd.Timestamp("2021-06-30")),
    ("LIVE      2026-07-11 -> now",  pd.Timestamp("2026-07-11"), pd.Timestamp("2026-09-06")),
]


# ─── shared portfolio helpers ─────────────────────────────────────────────────
class Book:
    def __init__(self, cash):
        self.cash = cash
        self.pos = {}          # sym -> {qty, cost, comm, entry_i}
        self.tax = 0.0
        self.sells = 0

    def sell(self, sym, p):
        h = self.pos.pop(sym)
        proceeds = h["qty"] * p
        sc = proceeds * COMMISSION
        gain = proceeds - h["cost"] - h["comm"] - sc
        t = max(0.0, gain * TAX_RATE)
        self.cash += proceeds - sc - t
        self.tax += t
        self.sells += 1

    def buy(self, sym, p, alloc, i):
        qty = int(alloc / (p * (1 + COMMISSION)))
        if qty < 1:
            return False
        bc = qty * p * COMMISSION
        self.cash -= qty * p + bc
        self.pos[sym] = {"qty": qty, "cost": qty * p, "comm": bc, "entry_i": i}
        return True

    def value(self, prices):
        return self.cash + sum(h["qty"] * prices[s] for s, h in self.pos.items()
                               if not np.isnan(prices[s]))

    def liquidate(self, prices):
        for sym in list(self.pos.keys()):
            p = prices[sym]
            if not np.isnan(p):
                self.sell(sym, p)


def stats(vals, days):
    v = pd.Series(vals, index=days)
    dd = ((v - v.cummax()) / v.cummax()).min() * 100
    return v, dd


# ─── engine 1: monthly momentum (generalized) ────────────────────────────────
def run_momentum(px, bull_series, days, month_firsts, regime_days, *,
                 top_n, keep_rank, sector_cap=None, vol_adj=False,
                 lb=252, cash0=INITIAL_CASH):
    bk = Book(cash0)
    bull_state = True
    rets = px.pct_change()
    values = []

    def scores(i):
        mom = (px.iloc[i - 21] / px.iloc[i - lb] - 1).dropna()
        if vol_adj:
            vol = rets.iloc[max(0, i - 126):i].std() * np.sqrt(252)
            vol = vol.reindex(mom.index).replace(0, np.nan)
            mom = (mom / vol).dropna()
        return mom.sort_values(ascending=False)

    def buy_slots(i, prices):
        slots = top_n - len(bk.pos)
        if slots <= 0 or i < 252:
            return
        ranked = scores(i)
        sec_count = {}
        for s in bk.pos:
            sec_count[SECTOR.get(s, "?")] = sec_count.get(SECTOR.get(s, "?"), 0) + 1
        per_slot = bk.cash * 0.98 / slots
        filled = 0
        for s in ranked.index:
            if filled >= slots:
                break
            if s in bk.pos or np.isnan(prices[s]):
                continue
            sec = SECTOR.get(s, "?")
            if sector_cap and sec_count.get(sec, 0) >= sector_cap:
                continue
            if bk.buy(s, prices[s], per_slot, i):
                sec_count[sec] = sec_count.get(sec, 0) + 1
                filled += 1

    for i, day in enumerate(days):
        bk.cash *= (1 + mmf_daily(day))
        prices = px.iloc[i]

        if day in regime_days and i >= 252:
            prev = bull_state
            bull_state = bool(bull_series.iloc[i])
            if prev and not bull_state:
                bk.liquidate(prices)
            elif not prev and bull_state and day not in month_firsts:
                buy_slots(i, prices)

        if day in month_firsts and i >= 252 and bull_state:
            ranked = scores(i)
            rank_of = {s: r + 1 for r, s in enumerate(ranked.index)}
            for sym in list(bk.pos.keys()):
                r = rank_of.get(sym)
                if r is not None and r <= keep_rank:
                    continue
                if not np.isnan(prices[sym]):
                    bk.sell(sym, prices[sym])
            buy_slots(i, prices)

        values.append(bk.value(prices))

    bk.liquidate(px.iloc[-1])
    return bk, values


# ─── engine 2: daily RSI(2) mean reversion ───────────────────────────────────
def rsi2(px):
    delta = px.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    # Wilder smoothing, period 2
    au = up.ewm(alpha=1 / 2, adjust=False).mean()
    ad = dn.ewm(alpha=1 / 2, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def run_meanrev(px, bull_series, days, regime_days, *,
                max_pos=10, entry_rsi=10.0, exit_rsi=65.0, max_hold=7,
                cash0=INITIAL_CASH):
    bk = Book(cash0)
    bull_state = True
    r2 = rsi2(px)
    sma200 = px.rolling(200).mean()
    values = []

    for i, day in enumerate(days):
        bk.cash *= (1 + mmf_daily(day))
        prices = px.iloc[i]

        if day in regime_days and i >= 252:
            bull_state = bool(bull_series.iloc[i])
        if not bull_state and bk.pos:
            bk.liquidate(prices)

        if i >= 252:
            # exits: rsi recovered or timed out
            for sym in list(bk.pos.keys()):
                p = prices[sym]
                if np.isnan(p):
                    continue
                held = i - bk.pos[sym]["entry_i"]
                r = r2.iloc[i].get(sym, np.nan)
                if (not np.isnan(r) and r > exit_rsi) or held >= max_hold:
                    bk.sell(sym, p)

            # entries (bull only): deepest oversold first, above own SMA200
            if bull_state:
                slots = max_pos - len(bk.pos)
                if slots > 0:
                    row = r2.iloc[i]
                    cands = []
                    for s in px.columns:
                        if s in bk.pos:
                            continue
                        p, r, m = prices[s], row.get(s, np.nan), sma200.iloc[i].get(s, np.nan)
                        if np.isnan(p) or np.isnan(r) or np.isnan(m):
                            continue
                        if r < entry_rsi and p > m:
                            cands.append((r, s))
                    cands.sort()
                    pv = bk.value(prices)
                    for r, s in cands[:slots]:
                        alloc = min(pv / max_pos, bk.cash * 0.95)
                        if alloc < 200:
                            break
                        bk.buy(s, prices[s], alloc, i)

        values.append(bk.value(prices))

    bk.liquidate(px.iloc[-1])
    return bk, values


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    print("Fetching data (2015 -> 2026-09)...")
    closes = fetch_closes(UNIVERSE)
    bench = fetch_closes(list(BENCH.values()))
    print(f"  {len(closes)} stocks fetched")

    spx_full = bench["^GSPC"]
    ndx_full = bench["^NDX"]
    days_full = spx_full.index
    px_full = pd.DataFrame(closes).sort_index().ffill().reindex(days_full).ffill()

    for pname, p_start, p_end in PERIODS:
        sim_days = days_full[(days_full >= p_start) & (days_full <= p_end)]
        if len(sim_days) < 5:
            continue
        i0 = list(days_full).index(sim_days[0])
        px = px_full.iloc[max(0, i0 - 260):list(days_full).index(sim_days[-1]) + 1]
        spx = spx_full.reindex(px.index).ffill()
        ndx = ndx_full.reindex(px.index).ffill()
        days = px.index
        in_sim = [d for d in days if p_start <= d <= p_end]
        n_years = max(len(in_sim) / 252, 0.05)

        bull_hybrid = (spx > spx.rolling(200).mean()) & (ndx > ndx.rolling(200).mean())

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

        print(f"\n{'='*108}")
        print(f"  {pname}")
        print(f"  Benchmarks net: SPX {bench_net['S&P 500']:+.2f}%  NDX {bench_net['NASDAQ-100']:+.2f}%  DJI {bench_net['Dow Jones']:+.2f}%")
        print(f"{'='*108}")
        print(f"  {'Strategy':<44} {'Return':>9} {'CAGR':>8} {'MaxDD':>8} {'Sells':>6} {'Tax $':>9} {'vs SPX':>8}")
        print(f"  {'-'*100}")

        sim_slice = slice(len(days) - len(in_sim), len(days))

        def report(label, bk, values):
            vals = values[sim_slice.start:]
            v, dd = stats(vals, in_sim)
            ret = (bk.cash / INITIAL_CASH - 1) * 100
            cagr = ((bk.cash / INITIAL_CASH) ** (1 / n_years) - 1) * 100
            a = ret - bench_net["S&P 500"]
            print(f"  {label:<44} {ret:>+8.2f}% {cagr:>+7.2f}% {dd:>7.1f}% "
                  f"{bk.sells:>6} {bk.tax:>9,.0f} {a:>+7.2f}%")
            return ret

        # CUR — current live strategy
        bk, v = run_momentum(px, bull_hybrid, days, month_firsts, weekly,
                             top_n=5, keep_rank=12)
        report("CUR  Top5 buf12 (current live)", bk, v)

        # A1 — diversified EW
        bk, v = run_momentum(px, bull_hybrid, days, month_firsts, weekly,
                             top_n=10, keep_rank=20, sector_cap=2)
        report("A1   Top10 buf20 EW, sector-cap 2", bk, v)

        # A2 — vol-adjusted momentum + sector cap
        bk, v = run_momentum(px, bull_hybrid, days, month_firsts, weekly,
                             top_n=10, keep_rank=20, sector_cap=2, vol_adj=True)
        report("A2   Top10 buf20 vol-adj, sector-cap 2", bk, v)

        # A3
        bk, v = run_momentum(px, bull_hybrid, days, month_firsts, weekly,
                             top_n=8, keep_rank=16, sector_cap=2, vol_adj=True)
        report("A3   Top8 buf16 vol-adj, sector-cap 2", bk, v)

        # MR — daily mean reversion
        bk, v = run_meanrev(px, bull_hybrid, days, weekly)
        report("MR   RSI(2)<10 dip-buy, 10 pos, daily", bk, v)

        # BL — 60% A2 + 40% MR
        bk_a, v_a = run_momentum(px, bull_hybrid, days, month_firsts, weekly,
                                 top_n=10, keep_rank=20, sector_cap=2,
                                 vol_adj=True, cash0=INITIAL_CASH * 0.6)
        bk_m, v_m = run_meanrev(px, bull_hybrid, days, weekly,
                                cash0=INITIAL_CASH * 0.4)
        comb_vals = [a + b for a, b in zip(v_a, v_m)]
        comb = Book(bk_a.cash + bk_m.cash)
        comb.tax = bk_a.tax + bk_m.tax
        comb.sells = bk_a.sells + bk_m.sells
        report("BL   60% A2 + 40% MR", comb, comb_vals)


if __name__ == "__main__":
    main()
