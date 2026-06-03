"""
TA-125 Backtest v9 — Refining AB (tp=65%, hold=40d)
Period: April 2021 - March 2026 (5 years)

AB baseline: +86.89%, 96 trail stops, 10 TP exits, 20 signal exits

Three targeted improvements:
  1. No-TP in BULL  — remove take-profit ceiling in BULL; only ATR trail + regime exit
  2. Wide ATR trail  — BULL ATR multiplier 3.5 → 4.5; ride through normal corrections
  3. Risk-Parity sizing — position size = fixed_risk / (ATR * mult), not flat pos_pct
     Stable stocks get bigger positions, volatile stocks smaller.
     Risk per trade = 1.5% of portfolio value.

Tests AB + each improvement alone + combinations (7 variants).
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

# ─── Regime params: AB baseline + optional modifications ──────────────────────
# (atr_mult, init_stop, tp_pct, min_bull_buy, min_bear_exit, max_pos, pos_pct, min_hold, rs_min, mean_rev)
AB_PARAMS = {
    "BULL":    (3.5, 0.08, 0.65, 4, 4, 10, 0.12, 40, 3.0, False),
    "NEUTRAL": (2.0, 0.06, 0.30, 4, 3,  5, 0.08, 20, 0.0,  True),
    "BEAR":    (1.0, 0.05, 0.10, 9, 2,  0, 0.05,  0, 0.0, False),
}

RISK_PER_TRADE_PCT = 0.015   # 1.5% portfolio at risk per trade (for risk-parity sizing)
MAX_SINGLE_POS_PCT = 0.20    # cap: never more than 20% of portfolio in one stock

def make_params(no_tp_bull=False, wide_atr=False):
    """Build regime params with optional modifications over AB baseline."""
    p = {}
    for regime, base in AB_PARAMS.items():
        (atr_mult, init_stop, tp_pct, min_bull_buy, min_bear_exit,
         max_pos, pos_pct, min_hold, rs_min, mean_rev) = base
        if regime == "BULL":
            if no_tp_bull: tp_pct   = 999.0   # effectively infinite — no TP exit in BULL
            if wide_atr:   atr_mult = 4.5      # wider trail, ride corrections
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

def relative_strength(stock_df, index_df, day):
    sh = stock_df[stock_df.index <= day]["Close"]
    ih = index_df[index_df.index <= day]["Close"]
    if len(sh) < RS_LOOKBACK + 1 or len(ih) < RS_LOOKBACK + 1: return None
    return round((float(sh.iloc[-1])/float(sh.iloc[-RS_LOOKBACK]) - 1)*100 -
                 (float(ih.iloc[-1])/float(ih.iloc[-RS_LOOKBACK]) - 1)*100, 2)

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

def risk_parity_qty(portfolio_val, price, atr, atr_mult, init_stop,
                    cash, risk_pct=RISK_PER_TRADE_PCT, cap_pct=MAX_SINGLE_POS_PCT):
    """
    Size position so that if stop is hit, we lose risk_pct of portfolio.
    stop_distance = max(atr * atr_mult, price * init_stop)
    qty = (portfolio * risk_pct) / stop_distance
    """
    if atr is None or atr <= 0:
        stop_dist = price * init_stop
    else:
        stop_dist = max(atr * atr_mult, price * init_stop)
    if stop_dist <= 0: return 0
    risk_amount = portfolio_val * risk_pct
    qty = int(risk_amount / stop_dist)
    # Cap: never more than cap_pct of portfolio in one position
    max_by_cap  = int(portfolio_val * cap_pct / (price * (1 + COMMISSION)))
    max_by_cash = int(cash * 0.95 / (price * (1 + COMMISSION)))
    return min(qty, max_by_cap, max_by_cash)


# ─── Portfolio backtest ────────────────────────────────────────────────────────
def run_backtest(valid_stocks, all_data, index_df, regime_params,
                 use_risk_parity, label):
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

        for sym, pos in positions.items():
            if sym in all_data and day in all_data[sym].index:
                cp = float(all_data[sym].loc[day, "Close"])
                if cp > pos.trail_high: pos.trail_high = cp

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
                pct = chg*100; peak = (pos.trail_high/pos.avg_cost-1)*100
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
                note=f"BOND_LIQ", regime=regime, category="BOND"))
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
                bp  = float(all_data[sym].loc[day, "Close"])
                if bp <= 0: continue
                df_sl = all_data[sym][all_data[sym].index <= day].tail(260)
                atr   = compute_atr(df_sl)

                pv = cash + sum(
                    p.quantity * float(all_data[p.symbol].loc[day,"Close"])
                    for p in positions.values()
                    if p.symbol in all_data and day in all_data[p.symbol].index
                )

                if use_risk_parity:
                    qty = risk_parity_qty(pv, bp, atr, atr_mult, init_stop, cash)
                else:
                    qty = int(min(pv * pos_pct, cash * 0.95) / (bp * (1 + COMMISSION)))

                if qty < 1: continue
                outlay = qty * bp * (1 + COMMISSION)
                if outlay > cash: continue
                bc = qty * bp * COMMISSION
                cash -= outlay
                positions[sym] = Position(symbol=sym, quantity=qty, avg_cost=bp,
                    buy_commission=bc, trail_high=bp,
                    regime_at_buy=regime, entry_day_idx=day_idx)
                tag = f"RP-{qty}sh" if use_risk_parity else f"RS+{rs:.1f}%"
                trades.append(Trade(date=day_str, symbol=sym, side="BUY",
                    quantity=qty, price=round(bp,3), commission=round(bc,2),
                    note=f"[{regime}|{tag}] " + "; ".join(sig["bull_details"][:2]),
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
    last_day  = trading_days[-1]
    stock_mkt = 0.0
    open_pos  = []
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

    tp_exits   = sum(1 for t in sell_stock if "TAKE_PROFIT" in t.note)
    stop_exits = sum(1 for t in sell_stock if "TRAIL_STOP"  in t.note)
    bear_exits = sum(1 for t in sell_stock if "BEAR_EXIT"   in t.note)
    sig_exits  = sum(1 for t in sell_stock if "BEARISH"     in t.note)

    return {
        "label": label, "total_val": total_val, "total_ret": total_ret,
        "cagr": ((1+total_ret/100)**(1/5)-1)*100,
        "avg_util": sum(daily_util)/len(daily_util) if daily_util else 0,
        "win_rate": wr, "avg_win": avg_win, "avg_loss": avg_loss,
        "n_sells": len(sell_stock), "bond_net": bond_net, "total_tax": total_tax,
        "tp_exits": tp_exits, "stop_exits": stop_exits,
        "bear_exits": bear_exits, "sig_exits": sig_exits,
        "open_pos": open_pos, "cash": cash, "stock_mkt": stock_mkt, "bond_mkt": bond_mkt,
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
            gain     = INITIAL_CASH*(p1/p0 - 1)
            index_bh_net = (gain - max(0, gain*TAX_RATE)) / INITIAL_CASH * 100

    all_data: dict = {}
    for sym in TA125_UNIVERSE:
        try:
            df = yf.Ticker(sym).history(start=FETCH_START, end=FETCH_END)
            if df.empty or len(df) < 50: continue
            df.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in df.index])
            all_data[sym] = df
        except Exception: pass

    valid = list(all_data.keys())
    print(f"  {len(valid)} stocks, index B&H net {index_bh_net:+.1f}%\n")

    W = 124

    # ── Variant table ──────────────────────────────────────────────────────────
    variants = [
        # (label, no_tp, wide_atr, risk_parity)
        ("AB   Baseline   (tp=65%, ATR=3.5x, flat sizing)",     False, False, False),
        ("1    No-TP Bull  (no cap in BULL, ATR=3.5x, flat)",    True,  False, False),
        ("2    Wide ATR    (tp=65%,  ATR=4.5x, flat sizing)",    False, True,  False),
        ("3    Risk-Parity (tp=65%,  ATR=3.5x, RP sizing)",      False, False, True ),
        ("12   No-TP + Wide ATR          (flat sizing)",         True,  True,  False),
        ("13   No-TP + Risk-Parity       (ATR=3.5x)",            True,  False, True ),
        ("23   Wide ATR + Risk-Parity    (tp=65%)",              False, True,  True ),
        ("123  All three  (no-TP, ATR=4.5x, RP sizing)",         True,  True,  True ),
    ]

    print("=" * W)
    print("  TA-125 BACKTEST v9 — REFINING AB STRATEGY")
    print(f"  AB baseline: tp=65%, hold=40d, 10pos/12% | Improvements: No-TP, Wide-ATR, Risk-Parity")
    print("=" * W)
    print(f"\n  Running {len(variants)} variants...\n")

    results = []
    for label, no_tp, wide_atr, rp in variants:
        params = make_params(no_tp_bull=no_tp, wide_atr=wide_atr)
        r = run_backtest(valid, all_data, index_df, params, rp, label)
        if r:
            results.append(r)
            print(f"  {label:<56}  ret={r['total_ret']:>+7.2f}%  "
                  f"wr={r['win_rate']:>3.0f}%  R/R={r['rr']:.2f}  "
                  f"sells={r['n_sells']}  TP={r['tp_exits']} stop={r['stop_exits']} "
                  f"bear={r['bear_exits']} sig={r['sig_exits']}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  FINAL COMPARISON  |  April 2021 - March 2026  |  NIS 100,000")
    print(f"  {'-'*W}")
    print(f"  {'Variant':<56}  {'Return':>9}  {'CAGR':>7}  {'Util':>6}  "
          f"{'Win%':>6}  {'R/R':>5}  {'Tax NIS':>9}  {'Sells':>6}")
    print(f"  {'-'*W}")

    for name, ret, cagr, ut, wr in [
        ("v4  Adaptive Regime (no bonds)",   66.16, 10.7,  33, 60),
        ("v8-AB  tp=65%, hold=40d",          86.89, 13.32, 68, 54),
    ]:
        print(f"  {name:<56}  {ret:>+8.2f}%  {cagr:>+6.1f}%  {ut:>5.0f}%  "
              f"{wr:>5.0f}%  {'':>5}  {'':>9}  {'':>6}")

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

    # ── Best drill-down + full comparison progression ─────────────────────────
    best = max(results, key=lambda x: x["total_ret"])
    print(f"  BEST VARIANT: {best['label']}")
    print(f"  {'─'*W}")
    print(f"  Portfolio Value:        NIS {best['total_val']:>12,.2f}")
    print(f"  Total Return (5yr):         {best['total_ret']:>+10.2f}%")
    print(f"  CAGR:                       {best['cagr']:>+10.2f}%")
    print(f"  Avg Capital Utilization:    {best['avg_util']:>9.1f}%")
    print(f"  Win Rate / R/R:             {best['win_rate']:>8.0f}% / {best['rr']:.2f}x")
    print(f"  Avg Win:               NIS {best['avg_win']:>+10,.0f}")
    print(f"  Avg Loss:              NIS {best['avg_loss']:>+10,.0f}")
    print(f"  Exit breakdown:  TP={best['tp_exits']}  Stop={best['stop_exits']}  "
          f"Bear={best['bear_exits']}  Signal={best['sig_exits']}")
    print(f"  Total Tax Paid:         NIS {best['total_tax']:>12,.2f}")
    print(f"  Bond Net P&L:           NIS {best['bond_net']:>+10,.2f}")

    if best["open_pos"]:
        print(f"\n  Open Positions at 2026-03-31:")
        for sym, qty, avg, last, mv, up, upct in best["open_pos"]:
            print(f"    {sym:<10}  qty={qty}  avg={avg:,.0f}  last={last:,.0f}  "
                  f"MV=NIS{mv:,.0f}  PnL={up:+,.0f} ({upct:+.1f}%)")

    # ── Full strategy progression ──────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  FULL STRATEGY EVOLUTION  |  NIS 100,000 → ?")
    print(f"  {'─'*W}")
    progression = [
        ("v4   Adaptive Regime (no bonds)",              66.16,  10.7),
        ("v5   + Bond rotation",                         67.37,  10.85),
        ("v6a  Full universe + bonds",                   68.40,  10.99),
        ("v8-AB  tp=65%, hold=40d, flat sizing",         86.89,  13.32),
        (f"v9-{best['label'][:3]}  {best['label'][5:].strip()}", best["total_ret"], best["cagr"]),
        ("B&H TA-125 (net of 25% tax)",                 105.13,  15.5),
    ]
    for name, ret, cagr in progression:
        bar = "#" * int(ret / 2)
        print(f"  {name:<50}  {ret:>+7.2f}%  CAGR {cagr:>+5.1f}%  {bar}")
    print(f"  {'=' * W}\n")


if __name__ == "__main__":
    run()
