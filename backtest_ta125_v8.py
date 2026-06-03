"""
TA-125 Backtest v8 — Strategy Parameter Optimization
Period: April 2021 - March 2026 (5 years)

Tests 3 approaches + combinations (7 variants total):
  A  Big Winners     — raise take-profit threshold (35%→60%)
  B  Patient Hold    — longer minimum hold period (15d→40d in BULL)
  C  Concentrated    — fewer positions, larger size (10 slots→4, 12%→22%)
  AB A+B together
  AC A+C together
  BC B+C together
  ABC all three

Baseline: v6a (full 23-stock universe + bond rotation)
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass
import yfinance as yf
import pandas as pd
import ta

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
INITIAL_CASH      = 100_000.0
COMMISSION        = 0.0008
TAX_RATE          = 0.25
RS_LOOKBACK       = 63
ETF_COMMISSION    = 0.0005

INDEX_TICKER      = "^TA125.TA"
SIM_START         = pd.Timestamp("2021-04-01")
SIM_END           = pd.Timestamp("2026-03-31")
FETCH_START       = "2020-01-01"
FETCH_END         = "2026-04-01"

BOND_ANNUAL_YIELD = {2021: 0.015, 2022: 0.010, 2023: 0.043, 2024: 0.040, 2025: 0.038, 2026: 0.038}

MIN_CASH_RESERVE_PCT = 0.05

REGIME_ETF_ALLOC = {
    "BULL":    (0.00, 0.00),
    "NEUTRAL": (0.00, 0.40),
    "BEAR":    (0.00, 0.60),
}

TA125_UNIVERSE = [
    "POLI.TA","LUMI.TA","DSCT.TA","MZRH.TA","FIBI.TA",
    "NICE.TA","CAMT.TA","TSEM.TA","NVMI.TA","SPNS.TA","ITRN.TA",
    "ESLT.TA","TEVA.TA","ICL.TA","BEZQ.TA",
    "SKBN.TA","RSEL.TA",
    "PHNX.TA","HARL.TA","MGDL.TA","MNRN.TA",
    "AZRG.TA","AMOT.TA","ALHE.TA","ELCO.TA",
    "ENLT.TA","DLEKG.TA","BAZAN.TA","ILCO.TA",
]

# Baseline regime params (from v4):
# (atr_mult, init_stop, tp_pct, min_bull_buy, min_bear_exit, max_pos, pos_pct, min_hold, rs_min, mean_rev)
BASE_PARAMS = {
    "BULL":    (3.5, 0.08, 0.35, 4, 4, 10, 0.12, 15, 3.0, False),
    "NEUTRAL": (2.0, 0.06, 0.12, 4, 3,  5, 0.08,  5, 0.0,  True),
    "BEAR":    (1.0, 0.05, 0.10, 9, 2,  0, 0.05,  0, 0.0, False),
}

def make_params(big_tp=False, patient=False, concentrated=False):
    """Generate REGIME_PARAMS for given combination of approaches."""
    p = {}
    for regime, base in BASE_PARAMS.items():
        (atr_mult, init_stop, tp_pct, min_bull_buy, min_bear_exit,
         max_pos, pos_pct, min_hold, rs_min, mean_rev) = base

        if big_tp:
            if regime == "BULL":    tp_pct  = 0.65
            if regime == "NEUTRAL": tp_pct  = 0.30

        if patient:
            if regime == "BULL":    min_hold = 40
            if regime == "NEUTRAL": min_hold = 20

        if concentrated:
            if regime == "BULL":
                max_pos  = 4
                pos_pct  = 0.22
                rs_min   = 5.0   # stricter RS filter when concentrated
            if regime == "NEUTRAL":
                max_pos  = 3
                pos_pct  = 0.18

        p[regime] = (atr_mult, init_stop, tp_pct, min_bull_buy, min_bear_exit,
                     max_pos, pos_pct, min_hold, rs_min, mean_rev)
    return p


# ─── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class Position:
    symbol: str; quantity: int; avg_cost: float
    buy_commission: float; trail_high: float
    regime_at_buy: str; entry_day_idx: int

@dataclass
class EtfHolding:
    ticker: str; quantity: float; avg_cost: float; buy_commission: float

@dataclass
class Trade:
    date: str; symbol: str; side: str; quantity: float; price: float
    commission: float; gross_pnl: float = 0.0; taxable_pnl: float = 0.0
    tax: float = 0.0; net_pnl: float = 0.0
    note: str = ""; regime: str = ""; category: str = "STOCK"


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _last(s):
    v = s.iloc[-1]; return float(v) if pd.notna(v) else None

def _prev(s):
    v = s.iloc[-2] if len(s) > 1 else None
    return float(v) if v is not None and pd.notna(v) else None

def compute_atr(df_slice, window=14):
    if len(df_slice) < window + 1: return None
    s = ta.volatility.average_true_range(df_slice["High"], df_slice["Low"], df_slice["Close"], window)
    v = s.iloc[-1]; return float(v) if pd.notna(v) else None

def classify_regime(index_df, day):
    hist = index_df[index_df.index <= day].tail(260)
    if len(hist) < 60: return "BULL"
    close = hist["Close"]
    high  = hist["High"] if "High" in hist.columns else close
    low   = hist["Low"]  if "Low"  in hist.columns else close
    sma200 = _last(ta.trend.sma_indicator(close, 200))
    sma50_s = ta.trend.sma_indicator(close, 50)
    adx    = _last(ta.trend.adx(high, low, close, 14))
    c      = float(close.iloc[-1])
    if sma200 is None: return "BULL"
    sna   = sma50_s.dropna()
    slope = (float(sna.iloc[-1]) - float(sna.iloc[-31])) / float(sna.iloc[-31]) if len(sna) >= 31 else 0.0
    if c > sma200 and slope > 0.008 and (adx or 0) > 22: return "BULL"
    if c < sma200 and slope < -0.008:                     return "BEAR"
    return "NEUTRAL"

def relative_strength(stock_df, index_df, day, lookback=RS_LOOKBACK):
    sh = stock_df[stock_df.index <= day]["Close"]
    ih = index_df[index_df.index <= day]["Close"]
    if len(sh) < lookback + 1 or len(ih) < lookback + 1: return None
    return round((float(sh.iloc[-1])/float(sh.iloc[-lookback]) - 1)*100 -
                 (float(ih.iloc[-1])/float(ih.iloc[-lookback]) - 1)*100, 2)

def bond_daily_return(day):
    return BOND_ANNUAL_YIELD.get(day.year, 0.040) / 252

def compute_signals(df_slice, min_bull, min_bear):
    if len(df_slice) < 30:
        return {"bullish":0,"bearish":0,"bias":"INSUFFICIENT",
                "bull_details":[],"bear_details":[],"bb_pct":None,"rsi":None}
    close=df_slice["Close"]; high=df_slice["High"]; low=df_slice["Low"]
    sma20=ta.trend.sma_indicator(close,20); sma50=ta.trend.sma_indicator(close,50)
    sma200=(ta.trend.sma_indicator(close,200) if len(df_slice)>=200
            else pd.Series([float("nan")]*len(df_slice),index=close.index))
    ema12=ta.trend.ema_indicator(close,12); ema26=ta.trend.ema_indicator(close,26)
    rsi14=ta.momentum.rsi(close,14)
    mo=ta.trend.MACD(close,12,26,9); ml=mo.macd(); ms=mo.macd_signal()
    so=ta.momentum.StochasticOscillator(high,low,close,14,smooth_window=3)
    sk=so.stoch(); sd=so.stoch_signal()
    bb=ta.volatility.BollingerBands(close,20,window_dev=2)
    bbu=bb.bollinger_hband(); bbl=bb.bollinger_lband()
    c=float(close.iloc[-1])
    vs20=_last(sma20); vs50=_last(sma50); vs200=_last(sma200)
    ve12=_last(ema12); ve26=_last(ema26); vrsi=_last(rsi14)
    vml=_last(ml); vms=_last(ms); pvml=_prev(ml); pvms=_prev(ms)
    vsk=_last(sk); vsd=_last(sd); vbbu=_last(bbu); vbbl=_last(bbl)
    bb_pct=((c-vbbl)/(vbbu-vbbl)) if vbbu and vbbl and (vbbu-vbbl)>0 else None
    bull,bear=[],[]
    if vrsi is not None:
        if vrsi<30:   bull.append(f"RSI oversold ({vrsi:.1f})")
        elif vrsi<40: bull.append(f"RSI low ({vrsi:.1f})")
        elif vrsi>70: bear.append(f"RSI overbought ({vrsi:.1f})")
        elif vrsi>60: bear.append(f"RSI high ({vrsi:.1f})")
    if all(x is not None for x in [vml,vms,pvml,pvms]):
        if vml>vms and pvml<=pvms: bull.append("MACD bullish xover")
        elif vml<vms and pvml>=pvms: bear.append("MACD bearish xover")
        (bull if vml>vms else bear).append(f"MACD {'above' if vml>vms else 'below'} signal")
    if vs20:  (bull if c>vs20  else bear).append(f"{'above' if c>vs20 else 'below'} SMA20")
    if vs50:  (bull if c>vs50  else bear).append(f"{'above' if c>vs50 else 'below'} SMA50")
    if vs200 and not pd.isna(vs200):
        (bull if c>vs200 else bear).append(f"{'above' if c>vs200 else 'below'} SMA200")
    if ve12 and ve26:
        (bull if ve12>ve26 else bear).append(f"EMA12 {'>' if ve12>ve26 else '<'} EMA26")
    if bb_pct is not None:
        if c<vbbl:        bull.append("Below lower BB")
        elif c>vbbu:      bear.append("Above upper BB")
        elif bb_pct<0.30: bull.append(f"Lower BB ({bb_pct*100:.0f}%)")
        elif bb_pct>0.70: bear.append(f"Upper BB ({bb_pct*100:.0f}%)")
    if vsk is not None and vsd is not None:
        if vsk<25 and vsd<25:   bull.append(f"Stoch oversold ({vsk:.0f})")
        elif vsk>75 and vsd>75: bear.append(f"Stoch overbought ({vsk:.0f})")
    nb,nb2=len(bull),len(bear)
    if nb>=min_bull and nb>nb2+1:    bias="BULLISH"
    elif nb2>=min_bear and nb2>nb+1: bias="BEARISH"
    else:                             bias="NEUTRAL"
    return {"bullish":nb,"bearish":nb2,"bias":bias,
            "rsi":round(vrsi,1) if vrsi else None,
            "close":c,"bb_pct":bb_pct,"bull_details":bull,"bear_details":bear}

def is_mean_reversion_entry(sig):
    rsi=sig.get("rsi") or 999; bp=sig.get("bb_pct")
    return bp is not None and rsi < 38 and bp < 0.25


# ─── Portfolio backtest (parameterized) ───────────────────────────────────────
def run_backtest(valid_stocks, all_data, index_df, regime_params, label):
    all_dates = set()
    for s in valid_stocks:
        if s in all_data: all_dates.update(all_data[s].index.tolist())
    if index_df is not None: all_dates.update(index_df.index.tolist())
    trading_days = sorted(d for d in all_dates if SIM_START <= d <= SIM_END)
    if not trading_days: return None

    cash          = INITIAL_CASH
    positions:    dict[str, Position]   = {}
    etf_holdings: dict[str, EtfHolding] = {}
    trades:       list[Trade]           = []
    total_tax     = 0.0
    daily_util    = []

    for day_idx, day in enumerate(trading_days):
        day_str = day.strftime("%Y-%m-%d")
        regime  = classify_regime(index_df, day) if index_df is not None else "BULL"
        (atr_mult, init_stop, tp_pct, min_bull_buy, min_bear_exit,
         max_pos, pos_pct, min_hold, rs_min, mean_rev) = regime_params[regime]
        _, bond_alloc = REGIME_ETF_ALLOC[regime]

        # Update trail highs
        for sym, pos in positions.items():
            if sym in all_data and day in all_data[sym].index:
                cp = float(all_data[sym].loc[day, "Close"])
                if cp > pos.trail_high: pos.trail_high = cp

        # Accrue bond
        if "BOND" in etf_holdings:
            etf_holdings["BOND"].quantity *= (1 + bond_daily_return(day))

        # ── STOCK SELLS ──────────────────────────────────────────────────────
        to_sell = []
        for sym, pos in list(positions.items()):
            if sym not in all_data or day not in all_data[sym].index: continue
            cp        = float(all_data[sym].loc[day, "Close"])
            chg       = (cp - pos.avg_cost) / pos.avg_cost
            hold_days = day_idx - pos.entry_day_idx
            (e_atr_mult, e_init_stop, e_tp_pct, _, e_min_bear,
             _, _, e_min_hold, _, _) = regime_params[pos.regime_at_buy]
            df_sl      = all_data[sym][all_data[sym].index <= day].tail(260)
            sig        = compute_signals(df_sl, 3, e_min_bear)
            atr        = compute_atr(df_sl)
            floor      = pos.avg_cost * (1 - e_init_stop)
            trail_stop = pos.trail_high - atr * e_atr_mult if atr else floor
            eff_stop   = max(trail_stop, floor)

            if cp <= eff_stop:
                pct = chg*100; peak = (pos.trail_high/pos.avg_cost - 1)*100
                atr_s = f"{atr:.1f}" if atr else "n/a"
                to_sell.append((sym, f"TRAIL_STOP ({pct:+.1f}%, peak+{peak:.1f}%)", sig))
            elif regime == "BEAR":
                to_sell.append((sym, "BEAR_EXIT", sig))
            elif chg >= e_tp_pct and hold_days >= e_min_hold and sig["bias"] in ("BEARISH","NEUTRAL"):
                to_sell.append((sym, f"TAKE_PROFIT ({chg*100:.1f}%, held {hold_days}d)", sig))
            elif hold_days >= e_min_hold and sig["bias"] == "BEARISH" and sig["bearish"] >= e_min_bear:
                to_sell.append((sym, f"BEARISH ({sig['bearish']}b, held {hold_days}d)", sig))

        for sym, reason, sig in to_sell:
            pos = positions[sym]
            sp  = float(all_data[sym].loc[day, "Close"])
            sc  = pos.quantity * sp * COMMISSION
            gp  = (sp - pos.avg_cost) * pos.quantity
            tx  = gp - pos.buy_commission - sc
            tax = max(0.0, tx * TAX_RATE)
            cash += pos.quantity * sp - sc - tax
            total_tax += tax
            trades.append(Trade(date=day_str, symbol=sym, side="SELL",
                quantity=pos.quantity, price=round(sp,3), commission=round(sc,2),
                gross_pnl=round(gp,2), taxable_pnl=round(tx,2),
                tax=round(tax,2), net_pnl=round(tx-tax,2),
                note=reason, regime=regime, category="STOCK"))
            del positions[sym]

        # ── BOND SELL ────────────────────────────────────────────────────────
        if bond_alloc == 0.0 and "BOND" in etf_holdings:
            h = etf_holdings["BOND"]
            gp = h.quantity - h.avg_cost
            sc = h.quantity * ETF_COMMISSION
            tx = gp - h.buy_commission - sc
            tax = max(0.0, tx * TAX_RATE)
            cash += h.quantity - sc - tax
            total_tax += tax
            trades.append(Trade(date=day_str, symbol="BOND", side="SELL",
                quantity=round(h.quantity,2), price=1.0,
                commission=round(sc,2), gross_pnl=round(gp,2),
                taxable_pnl=round(tx,2), tax=round(tax,2), net_pnl=round(tx-tax,2),
                note=f"BOND_LIQ (regime={regime})", regime=regime, category="BOND"))
            del etf_holdings["BOND"]

        # ── STOCK BUYS ───────────────────────────────────────────────────────
        open_slots = max_pos - len(positions)
        if open_slots > 0 and cash >= 500 and regime != "BEAR":
            candidates = []
            for sym in valid_stocks:
                if sym in positions: continue
                if sym not in all_data or day not in all_data[sym].index: continue
                df_sl = all_data[sym][all_data[sym].index <= day].tail(260)
                sig   = compute_signals(df_sl, min_bull_buy, 3)
                if mean_rev:
                    if not is_mean_reversion_entry(sig): continue
                    if sig["bias"] == "BEARISH": continue
                    candidates.append((sym, sig["bullish"]-sig["bearish"], 0.0, sig))
                else:
                    if sig["bias"] != "BULLISH" or sig["bullish"] < min_bull_buy: continue
                    rs = relative_strength(all_data[sym], index_df, day) if index_df is not None else 0.0
                    if rs is None or rs < rs_min: continue
                    candidates.append((sym, sig["bullish"]-sig["bearish"], rs, sig))

            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            for sym, score, rs, sig in candidates[:open_slots]:
                bp = float(all_data[sym].loc[day, "Close"])
                if bp <= 0: continue
                pv = cash + sum(
                    p.quantity * float(all_data[p.symbol].loc[day,"Close"])
                    for p in positions.values()
                    if p.symbol in all_data and day in all_data[p.symbol].index
                )
                qty = int(min(pv * pos_pct, cash * 0.95) / (bp * (1 + COMMISSION)))
                if qty < 1: continue
                outlay = qty * bp * (1 + COMMISSION)
                if outlay > cash: continue
                bc = qty * bp * COMMISSION
                cash -= outlay
                positions[sym] = Position(symbol=sym, quantity=qty, avg_cost=bp,
                    buy_commission=bc, trail_high=bp,
                    regime_at_buy=regime, entry_day_idx=day_idx)
                trades.append(Trade(date=day_str, symbol=sym, side="BUY",
                    quantity=qty, price=round(bp,3), commission=round(bc,2),
                    note=f"[{regime}] RS+{rs:.1f}% " + "; ".join(sig["bull_details"][:2]),
                    regime=regime, category="STOCK"))

        # ── BOND BUY ─────────────────────────────────────────────────────────
        pos_val  = sum(
            p.quantity * float(all_data[p.symbol].loc[day,"Close"])
            for p in positions.values()
            if p.symbol in all_data and day in all_data[p.symbol].index
        )
        bond_val = etf_holdings["BOND"].quantity if "BOND" in etf_holdings else 0.0
        port_val = cash + pos_val + bond_val
        deployable = max(0.0, cash - port_val * MIN_CASH_RESERVE_PCT)
        target_bond = port_val * bond_alloc

        if bond_alloc > 0 and deployable > 500 and bond_val < target_bond * 0.90:
            to_invest = min(target_bond - bond_val, deployable * 0.9)
            bc   = to_invest * ETF_COMMISSION
            cost = to_invest + bc
            if cost <= cash:
                cash -= cost
                if "BOND" in etf_holdings:
                    h = etf_holdings["BOND"]
                    h.avg_cost += to_invest; h.quantity += to_invest; h.buy_commission += bc
                else:
                    etf_holdings["BOND"] = EtfHolding("BOND", to_invest, to_invest, bc)
                trades.append(Trade(date=day_str, symbol="BOND", side="BUY",
                    quantity=round(to_invest,2), price=1.0, commission=round(bc,2),
                    note=f"[{regime}] Bond ~{BOND_ANNUAL_YIELD.get(day.year,0.04)*100:.1f}%/yr",
                    regime=regime, category="BOND"))

        daily_util.append((pos_val + bond_val) / (cash + pos_val + bond_val) * 100
                          if (cash + pos_val + bond_val) > 0 else 0)

    # ── Final valuation ───────────────────────────────────────────────────────
    last_day   = trading_days[-1]
    stock_mkt  = 0.0
    open_pos   = []
    for sym, pos in positions.items():
        df = all_data.get(sym)
        lp = float(df.loc[last_day,"Close"]) if df is not None and last_day in df.index else pos.avg_cost
        mv = pos.quantity * lp
        up = mv - (pos.quantity * pos.avg_cost + pos.buy_commission)
        stock_mkt += mv
        open_pos.append((sym, pos.quantity, pos.avg_cost, lp, mv, up, (lp/pos.avg_cost-1)*100))

    bond_mkt  = etf_holdings["BOND"].quantity if "BOND" in etf_holdings else 0.0
    total_val = cash + stock_mkt + bond_mkt
    total_ret = (total_val - INITIAL_CASH) / INITIAL_CASH * 100

    sell_stock = [t for t in trades if t.side=="SELL" and t.category=="STOCK"]
    wins       = [t for t in sell_stock if t.net_pnl > 0]
    losses     = [t for t in sell_stock if t.net_pnl <= 0]
    wr         = len(wins)/len(sell_stock)*100 if sell_stock else 0
    avg_win    = sum(t.net_pnl for t in wins)/len(wins)     if wins   else 0
    avg_loss   = sum(t.net_pnl for t in losses)/len(losses) if losses else 0
    bond_net   = sum(t.net_pnl for t in trades if t.side=="SELL" and t.category=="BOND")

    # Count exit reasons
    tp_exits   = sum(1 for t in sell_stock if "TAKE_PROFIT" in t.note)
    stop_exits = sum(1 for t in sell_stock if "TRAIL_STOP" in t.note)
    bear_exits = sum(1 for t in sell_stock if "BEAR_EXIT"  in t.note)
    sig_exits  = sum(1 for t in sell_stock if "BEARISH"    in t.note)

    return {
        "label": label, "total_val": total_val, "total_ret": total_ret,
        "cagr": ((1+total_ret/100)**(1/5)-1)*100,
        "avg_util": sum(daily_util)/len(daily_util) if daily_util else 0,
        "win_rate": wr, "avg_win": avg_win, "avg_loss": avg_loss,
        "n_sells": len(sell_stock), "bond_net": bond_net, "total_tax": total_tax,
        "tp_exits": tp_exits, "stop_exits": stop_exits,
        "bear_exits": bear_exits, "sig_exits": sig_exits,
        "open_pos": open_pos, "cash": cash,
        "rr": abs(avg_win/avg_loss) if avg_loss else 0,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def run():
    print("Fetching data...")
    index_df = None
    try:
        idx = yf.Ticker(INDEX_TICKER).history(start=FETCH_START, end=FETCH_END)
        if not idx.empty:
            idx.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in idx.index])
            index_df = idx
    except Exception: pass

    index_bh_net = bm_gross = 0.0
    if index_df is not None:
        sim = index_df[(index_df.index >= SIM_START) & (index_df.index <= SIM_END)]
        if len(sim) >= 2:
            p0, p1 = float(sim["Close"].iloc[0]), float(sim["Close"].iloc[-1])
            bm_gross = (p1/p0 - 1)*100
            gain = INITIAL_CASH*(p1/p0 - 1)
            index_bh_net = (gain - max(0, gain*TAX_RATE)) / INITIAL_CASH * 100

    all_data: dict = {}
    for sym in TA125_UNIVERSE:
        try:
            df = yf.Ticker(sym).history(start=FETCH_START, end=FETCH_END)
            if df.empty or len(df) < 50: continue
            df.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in df.index])
            all_data[sym] = df
        except Exception: pass

    valid = [s for s in all_data]
    print(f"  {len(valid)} stocks loaded, index B&H net {index_bh_net:+.1f}%\n")

    W = 120

    # ── Variant definitions ────────────────────────────────────────────────────
    variants = [
        ("Baseline  (v6a: tp=35%, hold=15d, 10pos/12%)",
         make_params(False, False, False)),
        ("A  Big Winners      (tp=65%, hold=15d, 10pos/12%)",
         make_params(True,  False, False)),
        ("B  Patient Hold     (tp=35%, hold=40d, 10pos/12%)",
         make_params(False, True,  False)),
        ("C  Concentrated     (tp=35%, hold=15d,  4pos/22%, RS>5%)",
         make_params(False, False, True)),
        ("AB Big Win + Patient (tp=65%, hold=40d, 10pos/12%)",
         make_params(True,  True,  False)),
        ("AC Big Win + Conc.  (tp=65%, hold=15d,  4pos/22%)",
         make_params(True,  False, True)),
        ("BC Patient + Conc.  (tp=35%, hold=40d,  4pos/22%)",
         make_params(False, True,  True)),
        ("ABC All three       (tp=65%, hold=40d,  4pos/22%)",
         make_params(True,  True,  True)),
    ]

    print("=" * W)
    print("  TA-125 BACKTEST v8 — STRATEGY PARAMETER OPTIMIZATION")
    print(f"  Full 23-stock universe + bond rotation | April 2021 - March 2026")
    print("=" * W)
    print(f"\n  Running {len(variants)} variants...\n")

    results = []
    for label, params in variants:
        r = run_backtest(valid, all_data, index_df, params, label)
        if r:
            results.append(r)
            print(f"  {label:<56}  ret={r['total_ret']:>+7.2f}%  "
                  f"tax=NIS{r['total_tax']:>7,.0f}  bond=NIS{r['bond_net']:>+6,.0f}  "
                  f"sells={r['n_sells']}  TP={r['tp_exits']} stop={r['stop_exits']}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  FINAL COMPARISON  |  April 2021 - March 2026  |  NIS 100,000")
    print(f"  {'-'*W}")
    print(f"  {'Variant':<56}  {'Return':>9}  {'CAGR':>7}  {'Util':>6}  "
          f"{'Win%':>6}  {'R/R':>5}  {'Tax':>9}  {'Sells':>6}")
    print(f"  {'-'*W}")

    # Prior baselines
    for name, ret, cagr, ut, wr in [
        ("v4  Adaptive Regime (stocks only, no bonds)", 66.16, 10.7, 33, 60),
        ("v5  + Bond rotation",                         67.37, 10.85, 67, 59),
    ]:
        print(f"  {name:<56}  {ret:>+8.2f}%  {cagr:>+6.1f}%  {ut:>5.0f}%  {wr:>5.0f}%  {'':>5}  {'':>9}  {'':>6}")

    best_ret = max(r["total_ret"] for r in results)
    for r in results:
        marker = " <-- BEST" if r["total_ret"] == best_ret else ""
        print(f"  {r['label']:<56}  {r['total_ret']:>+8.2f}%  {r['cagr']:>+6.1f}%  "
              f"{r['avg_util']:>5.0f}%  {r['win_rate']:>5.0f}%  {r['rr']:>5.2f}  "
              f"NIS{r['total_tax']:>6,.0f}  {r['n_sells']:>6}{marker}")

    print(f"  {'─'*W}")
    print(f"  {'Buy & Hold TA-125 (net of 25% tax)':<56}  "
          f"{index_bh_net:>+8.2f}%  {((1+index_bh_net/100)**(1/5)-1)*100:>+6.1f}%  "
          f"{'100':>5}%  {'  -':>6}  {'  -':>5}  {'':>9}  {'':>6}")
    print(f"  {'=' * W}\n")

    # ── Best result drill-down ─────────────────────────────────────────────────
    best = max(results, key=lambda x: x["total_ret"])
    print(f"  BEST VARIANT: {best['label']}")
    print(f"  {'─'*W}")
    print(f"  Portfolio Value:        NIS {best['total_val']:>12,.2f}")
    print(f"  Total Return (5yr net):     {best['total_ret']:>+10.2f}%")
    print(f"  CAGR:                       {best['cagr']:>+10.2f}%")
    print(f"  Avg Capital Utilization:    {best['avg_util']:>9.1f}%")
    print(f"  Win Rate:                   {best['win_rate']:>9.0f}%")
    print(f"  Avg Win / Loss:         NIS {best['avg_win']:>+7,.0f}  /  NIS {best['avg_loss']:>+7,.0f}")
    print(f"  R/R Ratio:                  {best['rr']:>9.2f}x")
    print(f"  Exits: TP={best['tp_exits']}  Stop={best['stop_exits']}  "
          f"Bear={best['bear_exits']}  Signal={best['sig_exits']}")
    print(f"  Total Tax Paid:         NIS {best['total_tax']:>12,.2f}")
    print(f"  Bond Net P&L:           NIS {best['bond_net']:>+10,.2f}")
    if best["open_pos"]:
        print(f"\n  Open Positions at 2026-03-31:")
        for sym, qty, avg, last, mv, up, upct in best["open_pos"]:
            print(f"    {sym:<10}  qty={qty}  avg={avg:,.0f}  last={last:,.0f}  "
                  f"MV=NIS{mv:,.0f}  PnL={up:+,.0f} ({upct:+.1f}%)")
    print()


if __name__ == "__main__":
    run()
