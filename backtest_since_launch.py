"""
Backtest מיום פתיחת הפרויקט: 25/04/2026 עד היום
אסטרטגיה: v9-13 (No-TP in BULL + Risk-Parity)
"""
from __future__ import annotations
import warnings, sys
from dataclasses import dataclass, field
from typing import Optional
import yfinance as yf
import pandas as pd
import ta

warnings.filterwarnings("ignore")

INITIAL_CASH  = 100_000.0
COMMISSION    = 0.0008
TAX_RATE      = 0.25
RS_LOOKBACK   = 63

INDEX_TICKER  = "^TA125.TA"
SIM_START     = pd.Timestamp("2026-04-25")
SIM_END       = pd.Timestamp("2026-05-20")
FETCH_START   = "2025-06-01"   # need enough history for SMA200 etc.
FETCH_END     = "2026-05-21"

BOND_ANNUAL_YIELD = {2026: 0.038}

TA125_UNIVERSE = [
    "POLI.TA","LUMI.TA","DSCT.TA","FIBI.TA",
    "NICE.TA","CAMT.TA","TSEM.TA","NVMI.TA",
    "ESLT.TA","TEVA.TA","ICL.TA","BEZQ.TA",
    "SKBN.TA","RSEL.TA","HARL.TA","MGDL.TA",
    "AZRG.TA","AMOT.TA","ALHE.TA","ELCO.TA",
    "ENLT.TA","DLEKG.TA","ILCO.TA",
]

# v9-13: No-TP in BULL + Risk-Parity
PARAMS = {
    "BULL":    dict(atr_mult=3.5, init_stop=0.08, tp=999.0, max_pos=10, pos_pct=0.12, min_hold=40, rs_min=3.0, mean_rev=False),
    "NEUTRAL": dict(atr_mult=2.0, init_stop=0.06, tp=0.30,  max_pos=4,  pos_pct=0.08, min_hold=20, rs_min=0.0, mean_rev=True),
    "BEAR":    dict(atr_mult=1.0, init_stop=0.05, tp=0.10,  max_pos=0,  pos_pct=0.05, min_hold=0,  rs_min=0.0, mean_rev=False),
}
BOND_ALLOC = {"BULL": 0.0, "NEUTRAL": 0.40, "BEAR": 0.60}
RISK_PER_TRADE = 0.015
MAX_POS_PCT    = 0.20

@dataclass
class Position:
    sym: str
    qty: float
    avg_cost: float
    entry_date: pd.Timestamp
    trail_high: float
    days_held: int = 0
    strategy: str = "TA"

def _last(s):
    v = s.dropna()
    return float(v.iloc[-1]) if len(v) else None

def compute_atr(close, high, low, w=14):
    s = ta.volatility.average_true_range(high, low, close, w)
    v = s.dropna()
    return float(v.iloc[-1]) if len(v) else None

def classify_regime(idx_df):
    c = idx_df["Close"]
    h = idx_df["High"]; l = idx_df["Low"]
    sma200 = _last(ta.trend.sma_indicator(c, min(200, len(c)-1)))
    sma50s = ta.trend.sma_indicator(c, min(50, len(c)-1)).dropna()
    adx    = _last(ta.trend.adx(h, l, c, 14))
    cur    = float(c.iloc[-1])
    slope  = (float(sma50s.iloc[-1]) - float(sma50s.iloc[-31])) / float(sma50s.iloc[-31]) if len(sma50s) >= 31 else 0
    if sma200 is None: return "NEUTRAL"
    if cur > sma200 and slope > 0.008 and (adx or 0) > 22: return "BULL"
    if cur < sma200 and slope < -0.008: return "BEAR"
    return "NEUTRAL"

def rs_score(df, idx_df):
    sh = df["Close"]; ih = idx_df["Close"]
    if len(sh) < RS_LOOKBACK+1 or len(ih) < RS_LOOKBACK+1: return None
    ih_aligned = ih[ih.index <= sh.index[-1]]
    if len(ih_aligned) < RS_LOOKBACK+1: return None
    return (float(sh.iloc[-1])/float(sh.iloc[-RS_LOOKBACK])-1)*100 - \
           (float(ih_aligned.iloc[-1])/float(ih_aligned.iloc[-RS_LOOKBACK])-1)*100

def risk_parity_qty(portfolio, price, atr, atr_mult, init_stop, cash):
    stop_dist = max(atr*atr_mult, price*init_stop) if atr else price*init_stop
    qty = (portfolio * RISK_PER_TRADE) / stop_dist
    max_cap  = portfolio * MAX_POS_PCT / (price * (1+COMMISSION))
    max_cash = cash * 0.95 / (price * (1+COMMISSION))
    return max(0, min(qty, max_cap, max_cash))

def buy_signal(sym, df_slice, idx_slice, regime):
    p = PARAMS[regime]
    c = df_slice["Close"]; h = df_slice["High"]; l = df_slice["Low"]
    sma20  = _last(ta.trend.sma_indicator(c, 20))
    sma50  = _last(ta.trend.sma_indicator(c, 50))
    rsi    = _last(ta.momentum.rsi(c, 14))
    bb     = ta.volatility.BollingerBands(c, 20, 2)
    bbl    = _last(bb.bollinger_lband()); bbu = _last(bb.bollinger_hband())
    bb_pct = ((float(c.iloc[-1])-bbl)/(bbu-bbl)) if bbu and bbl and bbu>bbl else None
    macd_o = ta.trend.MACD(c, 12, 26, 9)
    ml = _last(macd_o.macd()); ms = _last(macd_o.macd_signal())
    price  = float(c.iloc[-1])

    n_bull = sum([
        sma20 and price > sma20,
        sma50 and price > sma50,
        ml is not None and ms is not None and ml > ms,
        rsi is not None and rsi < 55,
        bb_pct is not None and bb_pct < 0.75,
    ])

    if regime == "BULL":
        rs = rs_score(df_slice, idx_slice)
        return n_bull >= 4 and (rs or 0) > p["rs_min"]
    elif regime == "NEUTRAL":
        return (rsi is not None and rsi < 38 and
                bb_pct is not None and bb_pct < 0.25)
    return False

# ─── Fetch data ───────────────────────────────────────────────
print("Fetching data...")
idx_raw = yf.Ticker(INDEX_TICKER).history(start=FETCH_START, end=FETCH_END)
idx_raw.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in idx_raw.index])

stock_data = {}
for sym in TA125_UNIVERSE:
    try:
        df = yf.Ticker(sym).history(start=FETCH_START, end=FETCH_END)
        if df.empty or len(df) < 60: continue
        df.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in df.index])
        stock_data[sym] = df
    except: pass

print(f"Loaded {len(stock_data)} stocks, {len(idx_raw)} index days")

# ─── Simulation ───────────────────────────────────────────────
trading_days = idx_raw.loc[SIM_START:SIM_END].index
print(f"Simulating {len(trading_days)} trading days: {trading_days[0].date()} to {trading_days[-1].date()}\n")

cash       = INITIAL_CASH
positions  = {}   # sym -> Position
bond_value = 0.0
total_tax  = 0.0
trades     = []

for day in trading_days:
    idx_slice = idx_raw.loc[:day]
    if len(idx_slice) < 60: continue
    regime = classify_regime(idx_slice)
    p = PARAMS[regime]

    portfolio_val = cash + bond_value + sum(
        pos.qty * float(stock_data[s]["Close"].loc[:day].iloc[-1])
        for s, pos in positions.items() if s in stock_data
    )

    # ── Bond allocation ───────────────────────────────────────
    target_bond = portfolio_val * BOND_ALLOC[regime]
    daily_bond_yield = BOND_ANNUAL_YIELD.get(day.year, 0.038) / 252
    bond_value *= (1 + daily_bond_yield)
    if bond_value < target_bond * 0.95 and cash > (target_bond - bond_value):
        move = min(target_bond - bond_value, cash * 0.5)
        bond_value += move; cash -= move
    elif bond_value > target_bond * 1.05:
        move = bond_value - target_bond
        cash += move; bond_value -= move

    # ── Update trail stops & check exits ─────────────────────
    for sym in list(positions.keys()):
        if sym not in stock_data: continue
        pos = positions[sym]
        df_s  = stock_data[sym].loc[:day]
        if df_s.empty: continue
        price = float(df_s["Close"].iloc[-1])
        atr   = compute_atr(df_s["Close"], df_s["High"], df_s["Low"])
        pos.trail_high = max(pos.trail_high, price)
        pos.days_held += 1

        trail_stop = max(
            pos.trail_high - (atr * p["atr_mult"]) if atr else pos.avg_cost * 0.85,
            pos.avg_cost * (1 - p["init_stop"])
        )
        tp_price = pos.avg_cost * (1 + p["tp"]) if p["tp"] < 10 else None
        n_bear_sig = 0
        c_ser = df_s["Close"]
        rsi_v = _last(ta.momentum.rsi(c_ser, 14))
        sma50_v = _last(ta.trend.sma_indicator(c_ser, 50))
        if rsi_v and rsi_v > 65: n_bear_sig += 1
        if sma50_v and price < sma50_v: n_bear_sig += 1
        if regime == "BEAR": n_bear_sig += 3

        exit_reason = None
        if price <= trail_stop and pos.days_held >= p["min_hold"]:
            exit_reason = "trail_stop"
        elif tp_price and price >= tp_price:
            exit_reason = "take_profit"
        elif n_bear_sig >= 4 and pos.days_held >= p["min_hold"]:
            exit_reason = "signal_exit"

        if exit_reason:
            gross  = pos.qty * price * (1 - COMMISSION)
            cost   = pos.qty * pos.avg_cost
            gain   = gross - cost
            tax    = max(gain * TAX_RATE, 0)
            net    = gross - tax
            total_tax += tax
            cash   += net
            trades.append(dict(sym=sym, date=day, exit=exit_reason,
                               entry=pos.avg_cost, exit_p=price,
                               pnl_pct=(price/pos.avg_cost-1)*100, gain_net=gain-tax))
            del positions[sym]

    # ── Buy signals ───────────────────────────────────────────
    if regime != "BEAR" and len(positions) < p["max_pos"]:
        candidates = []
        for sym, df_s in stock_data.items():
            if sym in positions: continue
            df_slice = df_s.loc[:day]
            if len(df_slice) < 60: continue
            if buy_signal(sym, df_slice, idx_slice, regime):
                rs = rs_score(df_slice, idx_slice) or 0
                candidates.append((sym, rs, df_slice))
        candidates.sort(key=lambda x: x[1], reverse=True)
        for sym, rs, df_slice in candidates[:p["max_pos"] - len(positions)]:
            price = float(df_slice["Close"].iloc[-1])
            atr   = compute_atr(df_slice["Close"], df_slice["High"], df_slice["Low"])
            qty   = risk_parity_qty(portfolio_val, price, atr, p["atr_mult"], p["init_stop"], cash)
            cost  = qty * price * (1 + COMMISSION)
            if qty > 0 and cost <= cash * 0.95:
                cash -= cost
                positions[sym] = Position(sym=sym, qty=qty, avg_cost=price,
                                          entry_date=day, trail_high=price)
                trades.append(dict(sym=sym, date=day, exit="BUY",
                                   entry=price, exit_p=None, pnl_pct=None, gain_net=None))

# ─── Final portfolio value (mark open positions to market) ────
last_day = trading_days[-1]
open_pnl = 0
open_positions_info = []
for sym, pos in positions.items():
    if sym not in stock_data: continue
    last_price = float(stock_data[sym]["Close"].loc[:last_day].iloc[-1])
    val = pos.qty * last_price
    cost = pos.qty * pos.avg_cost
    atr = compute_atr(stock_data[sym]["Close"].loc[:last_day],
                      stock_data[sym]["High"].loc[:last_day],
                      stock_data[sym]["Low"].loc[:last_day])
    p = PARAMS["BULL"]  # use current regime params for trail stop
    trail_stop = max(
        pos.trail_high - (atr * p["atr_mult"]) if atr else pos.avg_cost * 0.85,
        pos.avg_cost * (1 - p["init_stop"])
    )
    open_pnl += val - cost
    open_positions_info.append(dict(
        sym=sym, entry=round(pos.avg_cost,2),
        last=round(last_price,2),
        pnl_pct=round((last_price/pos.avg_cost-1)*100,2),
        trail_stop=round(trail_stop,2),
        days=pos.days_held
    ))

final_portfolio = cash + bond_value + sum(
    pos.qty * float(stock_data[s]["Close"].loc[:last_day].iloc[-1])
    for s, pos in positions.items() if s in stock_data
)

# ─── Index benchmark ──────────────────────────────────────────
idx_start_price = float(idx_raw.loc[SIM_START:].iloc[0]["Close"])
idx_end_price   = float(idx_raw.loc[:last_day].iloc[-1]["Close"])
index_bh_pct    = (idx_end_price / idx_start_price - 1) * 100
strategy_pct    = (final_portfolio / INITIAL_CASH - 1) * 100

# ─── Output ───────────────────────────────────────────────────
print("=" * 60)
print(f"  בקטסט: {SIM_START.date()} - {last_day.date()}")
print(f"  ({len(trading_days)} ימי מסחר, {len(trading_days)/252*12:.1f} חודשים)")
print("=" * 60)
print(f"\n  תשואת האסטרטגיה:     {strategy_pct:+.2f}%")
print(f"  תשואת ת\"א 125 (B&H): {index_bh_pct:+.2f}%")
print(f"  אלפא:                 {strategy_pct - index_bh_pct:+.2f}%")
print(f"\n  תיק סופי:    ₪{final_portfolio:,.0f}  (התחלה: ₪{INITIAL_CASH:,.0f})")
print(f"  מזומן:       ₪{cash:,.0f}")
print(f"  אגח:         ₪{bond_value:,.0f}")
print(f"  מס שולם:     ₪{total_tax:,.0f}")
print(f"  עסקאות:      {len([t for t in trades if t['exit']!='BUY'])}")

print(f"\n  פוזיציות פתוחות ({len(open_positions_info)}):")
if open_positions_info:
    for p in sorted(open_positions_info, key=lambda x: x["pnl_pct"], reverse=True):
        sign = "+" if p["pnl_pct"] >= 0 else ""
        print(f"    {p['sym']:<12} כניסה: ₪{p['entry']:,.0f}  כעת: ₪{p['last']:,.0f}  "
              f"({sign}{p['pnl_pct']:.1f}%)  Trail Stop: ₪{p['trail_stop']:,.0f}  ימים: {p['days']}")
else:
    print("    אין פוזיציות פתוחות")

print(f"\n  מסחר שבוצע:")
buys  = [t for t in trades if t["exit"] == "BUY"]
exits = [t for t in trades if t["exit"] != "BUY"]
print(f"    קניות:  {len(buys)}")
print(f"    מכירות: {len(exits)}")
if exits:
    winners = [t for t in exits if (t["pnl_pct"] or 0) > 0]
    print(f"    WIN rate: {len(winners)}/{len(exits)} = {len(winners)/len(exits)*100:.0f}%")

print("\n" + "=" * 60)
