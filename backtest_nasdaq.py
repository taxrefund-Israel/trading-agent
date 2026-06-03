"""
NASDAQ 100 Backtest — v9-13 Strategy (No-TP in BULL + Risk-Parity sizing)
Period: April 2021 - March 2026 (5 years)
Capital: $10,000 USD (FX changes ignored)

Strategy (best from TA-125 research):
  - Regime-aware: BULL / NEUTRAL / BEAR (SMA200 + SMA50 slope + ADX)
  - BULL:    trend entries (RS > 3%), NO take-profit ceiling, ATR trail 3.5x, hold≥40d
  - NEUTRAL: mean-reversion entries (RSI<38, BB<25%), TP 30%, hold≥20d + bonds 40%
  - BEAR:    no stock positions, bonds 60%
  - Sizing:  Risk-Parity — size = (portfolio * 1.5%) / (ATR * 3.5)
  - Tax:     25% capital gains (Israeli investor in US stocks)
  - Bonds:   simulated US Treasury yield by year

Comparison: strategy return vs NASDAQ 100 buy-and-hold (net of 25% tax)
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass
import yfinance as yf
import pandas as pd
import ta

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
INITIAL_CASH      = 10_000.0     # USD
COMMISSION        = 0.0008       # 0.08% (same as TA-125)
TAX_RATE          = 0.25         # 25% Israeli CGT on foreign investments
RS_LOOKBACK       = 63
ETF_COMMISSION    = 0.0005

INDEX_TICKER      = "^NDX"       # NASDAQ 100
SIM_START         = pd.Timestamp("2021-04-01")
SIM_END           = pd.Timestamp("2026-03-31")
FETCH_START       = "2020-01-01"
FETCH_END         = "2026-04-01"

# US Treasury approximate net annual yield by year
BOND_ANNUAL_YIELD = {
    2021: 0.015,   # avg 10yr ~1.5%
    2022: 0.010,   # rising rates hurt bond prices, net low
    2023: 0.045,   # 10yr peaked ~5%, avg ~4.5%
    2024: 0.042,   # ~4.2%
    2025: 0.043,   # ~4.3%
    2026: 0.043,
}

MIN_CASH_RESERVE_PCT = 0.05
RISK_PER_TRADE_PCT   = 0.015    # 1.5% of portfolio at risk per trade
MAX_SINGLE_POS_PCT   = 0.20     # max 20% in one position

REGIME_ETF_ALLOC = {
    "BULL":    (0.00, 0.00),
    "NEUTRAL": (0.00, 0.40),
    "BEAR":    (0.00, 0.60),
}

# v9-13 params: No-TP in BULL + Risk-Parity
REGIME_PARAMS = {
    # (atr_mult, init_stop, tp_pct, min_bull_buy, min_bear_exit, max_pos, pos_pct, min_hold, rs_min, mean_rev)
    "BULL":    (3.5, 0.08, 999.0, 4, 4, 10, 0.12, 40, 3.0, False),  # tp=999 = no cap
    "NEUTRAL": (2.0, 0.06, 0.30,  4, 3,  5, 0.08, 20, 0.0,  True),
    "BEAR":    (1.0, 0.05, 0.10,  9, 2,  0, 0.05,  0, 0.0, False),
}

# Screening params (for Phase 1 per-stock analysis)
SCREEN_MIN_BULL  = 4
SCREEN_MIN_BEAR  = 4
SCREEN_STOP_PCT  = 0.08
SCREEN_TP_PCT    = 0.35
SCREEN_ATR_MULT  = 3.0
SCREEN_MIN_HOLD  = 5

NASDAQ_UNIVERSE = [
    # Mega cap
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Semiconductors
    "AMD", "AVGO", "QCOM", "INTC", "MU", "AMAT", "LRCX", "KLAC", "MCHP", "TXN",
    # Software / Cloud
    "CRM", "ADBE", "ORCL", "CDNS", "SNPS", "ADP", "PAYX",
    # Cybersecurity
    "PANW", "CRWD", "FTNT",
    # Internet / Streaming
    "NFLX", "CMCSA",
    # Biotech / Healthcare
    "AMGN", "GILD", "REGN", "VRTX", "ISRG", "IDXX",
    # Consumer / Retail
    "COST", "SBUX", "MNST",
    # Fintech / Other
    "PYPL", "FISV", "PCAR", "FAST", "CSCO",
]


# ─── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class Position:
    symbol: str; quantity: float; avg_cost: float
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
    close  = hist["Close"]
    high   = hist["High"] if "High" in hist.columns else close
    low    = hist["Low"]  if "Low"  in hist.columns else close
    sma200 = _last(ta.trend.sma_indicator(close, 200))
    sma50s = ta.trend.sma_indicator(close, 50)
    adx    = _last(ta.trend.adx(high, low, close, 14))
    c      = float(close.iloc[-1])
    if sma200 is None: return "BULL"
    sna   = sma50s.dropna()
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

def risk_parity_qty(portfolio_val, price, atr, atr_mult, init_stop, cash):
    stop_dist = max(atr * atr_mult, price * init_stop) if atr and atr > 0 else price * init_stop
    if stop_dist <= 0: return 0
    risk_amount = portfolio_val * RISK_PER_TRADE_PCT
    qty = risk_amount / stop_dist                            # fractional shares allowed
    # Cap: no more than MAX_SINGLE_POS_PCT of portfolio
    max_by_cap  = portfolio_val * MAX_SINGLE_POS_PCT / (price * (1 + COMMISSION))
    max_by_cash = cash * 0.95 / (price * (1 + COMMISSION))
    return min(qty, max_by_cap, max_by_cash)


# ─── Phase 1: Per-stock screening ─────────────────────────────────────────────
def screen_stock(sym, df, index_bh_net):
    sim_days = [d for d in df.index if SIM_START <= d <= SIM_END]
    if len(sim_days) < 60: return None
    capital = 10_000.0; cash = capital; pos = None; trades = []

    for day_idx, day in enumerate(sim_days):
        df_sl = df[df.index <= day].tail(260)
        price = float(df.loc[day, "Close"])
        if pos and price > pos["trail_high"]: pos["trail_high"] = price
        if pos:
            chg = (price - pos["avg_cost"]) / pos["avg_cost"]
            hold = day_idx - pos["entry_idx"]
            atr  = compute_atr(df_sl)
            floor = pos["avg_cost"] * (1 - SCREEN_STOP_PCT)
            trail = pos["trail_high"] - atr * SCREEN_ATR_MULT if atr else floor
            eff   = max(trail, floor)
            reason = None
            if price <= eff: reason = f"STOP ({chg*100:+.1f}%)"
            elif hold >= SCREEN_MIN_HOLD:
                sig = compute_signals(df_sl, SCREEN_MIN_BULL, SCREEN_MIN_BEAR)
                if chg >= SCREEN_TP_PCT and sig["bias"] != "BULLISH":
                    reason = f"TP ({chg*100:+.1f}%)"
                elif sig["bias"] == "BEARISH" and sig["bearish"] >= SCREEN_MIN_BEAR:
                    reason = f"SIG ({chg*100:+.1f}%)"
            if reason:
                sc = pos["qty"] * price * COMMISSION
                gp = (price - pos["avg_cost"]) * pos["qty"]
                tx = gp - pos["buy_comm"] - sc
                cash += pos["qty"] * price - sc - max(0.0, tx * TAX_RATE)
                trades.append({"net": tx - max(0.0, tx * TAX_RATE)})
                pos = None
        if pos is None and cash >= 100:
            sig = compute_signals(df_sl, SCREEN_MIN_BULL, SCREEN_MIN_BEAR)
            if sig["bias"] == "BULLISH" and sig["bullish"] >= SCREEN_MIN_BULL:
                qty = cash * 0.95 / (price * (1 + COMMISSION))
                if qty >= 0.01:
                    bc = qty * price * COMMISSION
                    cash -= qty * price + bc
                    pos = {"qty": qty, "avg_cost": price, "buy_comm": bc,
                           "trail_high": price, "entry_idx": day_idx}

    if pos and sim_days:
        price = float(df.loc[sim_days[-1], "Close"])
        sc = pos["qty"] * price * COMMISSION
        gp = (price - pos["avg_cost"]) * pos["qty"]
        tx = gp - pos["buy_comm"] - sc
        cash += pos["qty"] * price - sc - max(0.0, tx * TAX_RATE)
        trades.append({"net": tx - max(0.0, tx * TAX_RATE)})

    ta_ret = (cash - capital) / capital * 100
    hist = df[(df.index >= SIM_START) & (df.index <= SIM_END)]
    bh_ret = 0.0
    if len(hist) >= 2:
        p0, p1 = float(hist["Close"].iloc[0]), float(hist["Close"].iloc[-1])
        bh_gain = capital * (p1/p0 - 1)
        bh_ret  = (bh_gain - max(0, bh_gain * TAX_RATE)) / capital * 100

    wins   = [t for t in trades if t["net"] > 0]
    losses = [t for t in trades if t["net"] <= 0]
    return {
        "symbol": sym, "ta_return": round(ta_ret,2), "bh_return": round(bh_ret,2),
        "alpha_idx": round(ta_ret - index_bh_net, 2),
        "trades": len(trades), "win_rate": round(len(wins)/len(trades)*100,1) if trades else 0,
    }


# ─── Portfolio backtest ────────────────────────────────────────────────────────
def run_backtest(valid_stocks, all_data, index_df):
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
    regime_days   = {"BULL": 0, "NEUTRAL": 0, "BEAR": 0}
    regime_log    = []
    last_regime   = None

    for day_idx, day in enumerate(trading_days):
        day_str = day.strftime("%Y-%m-%d")
        regime  = classify_regime(index_df, day) if index_df is not None else "BULL"
        (atr_mult, init_stop, tp_pct, min_bull_buy, min_bear_exit,
         max_pos, pos_pct, min_hold, rs_min, mean_rev) = REGIME_PARAMS[regime]
        _, bond_alloc = REGIME_ETF_ALLOC[regime]

        regime_days[regime] += 1
        if regime != last_regime:
            regime_log.append((day_str, regime)); last_regime = regime

        for sym, pos in positions.items():
            if sym in all_data and day in all_data[sym].index:
                cp = float(all_data[sym].loc[day, "Close"])
                if cp > pos.trail_high: pos.trail_high = cp

        if "BOND" in etf_holdings:
            etf_holdings["BOND"].quantity *= (1 + bond_daily_return(day))

        # ── SELLS ────────────────────────────────────────────────────────────
        to_sell = []
        for sym, pos in list(positions.items()):
            if sym not in all_data or day not in all_data[sym].index: continue
            cp        = float(all_data[sym].loc[day, "Close"])
            chg       = (cp - pos.avg_cost) / pos.avg_cost
            hold_days = day_idx - pos.entry_day_idx
            (e_atr, e_stop, e_tp, _, e_mb, _, _, e_mh, _, _) = REGIME_PARAMS[pos.regime_at_buy]
            df_sl  = all_data[sym][all_data[sym].index <= day].tail(260)
            sig    = compute_signals(df_sl, 3, e_mb)
            atr    = compute_atr(df_sl)
            floor  = pos.avg_cost * (1 - e_stop)
            trail  = pos.trail_high - atr * e_atr if atr else floor
            eff    = max(trail, floor)

            if cp <= eff:
                pct = chg*100; peak = (pos.trail_high/pos.avg_cost-1)*100
                to_sell.append((sym, f"TRAIL_STOP ({pct:+.1f}%, peak+{peak:.1f}%)", sig))
            elif regime == "BEAR":
                to_sell.append((sym, "BEAR_EXIT", sig))
            elif chg >= e_tp and hold_days >= e_mh and sig["bias"] in ("BEARISH","NEUTRAL"):
                to_sell.append((sym, f"TAKE_PROFIT ({chg*100:.1f}%, {hold_days}d)", sig))
            elif hold_days >= e_mh and sig["bias"] == "BEARISH" and sig["bearish"] >= e_mb:
                to_sell.append((sym, f"BEARISH ({sig['bearish']}b, {hold_days}d)", sig))

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
                quantity=round(pos.quantity,4), price=round(sp,4), commission=round(sc,4),
                gross_pnl=round(gp,4), taxable_pnl=round(tx,4),
                tax=round(tax,4), net_pnl=round(tx-tax,4),
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
                quantity=round(h.quantity,4), price=1.0, commission=round(sc,4),
                gross_pnl=round(gp,4), taxable_pnl=round(tx,4),
                tax=round(tax,4), net_pnl=round(tx-tax,4),
                note="BOND_LIQ", regime=regime, category="BOND"))
            del etf_holdings["BOND"]

        # ── STOCK BUYS ───────────────────────────────────────────────────────
        open_slots = max_pos - len(positions)
        if open_slots > 0 and cash >= 50 and regime != "BEAR":
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
                df_sl = all_data[sym][all_data[sym].index <= day].tail(260)
                atr   = compute_atr(df_sl)
                pv    = cash + sum(
                    p.quantity * float(all_data[p.symbol].loc[day,"Close"])
                    for p in positions.values()
                    if p.symbol in all_data and day in all_data[p.symbol].index
                )
                qty = risk_parity_qty(pv, bp, atr, atr_mult, init_stop, cash)
                if qty < 0.001: continue
                outlay = qty * bp * (1 + COMMISSION)
                if outlay > cash: continue
                bc = qty * bp * COMMISSION
                cash -= outlay
                positions[sym] = Position(symbol=sym, quantity=qty, avg_cost=bp,
                    buy_commission=bc, trail_high=bp,
                    regime_at_buy=regime, entry_day_idx=day_idx)
                trades.append(Trade(date=day_str, symbol=sym, side="BUY",
                    quantity=round(qty,4), price=round(bp,4), commission=round(bc,4),
                    note=f"[{regime}|RP|RS+{rs:.1f}%] " + "; ".join(sig["bull_details"][:2]),
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

        if bond_alloc > 0 and deployable > 50 and bond_val < target_bond * 0.90:
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
                    quantity=round(to_invest,4), price=1.0, commission=round(bc,4),
                    note=f"[{regime}] UST ~{BOND_ANNUAL_YIELD.get(day.year,0.04)*100:.1f}%/yr",
                    regime=regime, category="BOND"))

        daily_util.append((pos_val+bond_val)/(cash+pos_val+bond_val)*100
                          if (cash+pos_val+bond_val)>0 else 0)

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
    wins   = [t for t in sell_stock if t.net_pnl > 0]
    losses = [t for t in sell_stock if t.net_pnl <= 0]
    wr     = len(wins)/len(sell_stock)*100 if sell_stock else 0
    avg_w  = sum(t.net_pnl for t in wins)/len(wins)     if wins   else 0
    avg_l  = sum(t.net_pnl for t in losses)/len(losses) if losses else 0
    bond_n = sum(t.net_pnl for t in trades if t.side=="SELL" and t.category=="BOND")

    return {
        "total_val": total_val, "total_ret": total_ret,
        "cagr": ((1+total_ret/100)**(1/5)-1)*100,
        "avg_util": sum(daily_util)/len(daily_util) if daily_util else 0,
        "win_rate": wr, "avg_win": avg_w, "avg_loss": avg_l,
        "n_buys":  len([t for t in trades if t.side=="BUY"  and t.category=="STOCK"]),
        "n_sells": len(sell_stock), "bond_net": bond_n, "total_tax": total_tax,
        "tp_exits":   sum(1 for t in sell_stock if "TAKE_PROFIT" in t.note),
        "stop_exits": sum(1 for t in sell_stock if "TRAIL_STOP"  in t.note),
        "bear_exits": sum(1 for t in sell_stock if "BEAR_EXIT"   in t.note),
        "sig_exits":  sum(1 for t in sell_stock if "BEARISH"     in t.note),
        "open_pos": sorted(open_pos, key=lambda x: x[5], reverse=True),
        "cash": cash, "stock_mkt": stock_mkt, "bond_mkt": bond_mkt,
        "rr": abs(avg_w/avg_l) if avg_l else 0,
        "regime_days": regime_days, "regime_log": regime_log,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def run():
    print("Fetching NASDAQ data (60-90 sec)...")

    index_df = None
    try:
        idx = yf.Ticker(INDEX_TICKER).history(start=FETCH_START, end=FETCH_END)
        if not idx.empty:
            idx.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in idx.index])
            index_df = idx
            print(f"  Index {INDEX_TICKER}: {len(idx)} bars")
    except Exception as e:
        print(f"  Index fetch failed: {e}")

    index_bh_net = bm_gross = 0.0
    if index_df is not None:
        sim = index_df[(index_df.index >= SIM_START) & (index_df.index <= SIM_END)]
        if len(sim) >= 2:
            p0, p1 = float(sim["Close"].iloc[0]), float(sim["Close"].iloc[-1])
            bm_gross = (p1/p0 - 1)*100
            gain = INITIAL_CASH*(p1/p0 - 1)
            index_bh_net = (gain - max(0, gain*TAX_RATE)) / INITIAL_CASH * 100

    all_data: dict = {}
    for sym in NASDAQ_UNIVERSE:
        try:
            df = yf.Ticker(sym).history(start=FETCH_START, end=FETCH_END)
            if df.empty or len(df) < 100: continue
            df.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in df.index])
            all_data[sym] = df
        except Exception: pass
    valid = list(all_data.keys())
    print(f"  Stocks loaded: {len(valid)}\n")

    W = 122

    # ══ PHASE 1: Screening ═══════════════════════════════════════════════════
    print("=" * W)
    print("  PHASE 1 — PER-STOCK TA SCREENING  |  NASDAQ 100  |  April 2021 - March 2026")
    print(f"  Index B&H: gross {bm_gross:+.1f}%  net {index_bh_net:+.1f}%  (after 25% tax)")
    print("=" * W)
    print(f"\n  {'Symbol':<8}  {'TA%':>8}  {'B&H%':>9}  {'AlphaIdx':>10}  "
          f"{'Trades':>7}  {'Win%':>6}  {'Classification':>16}")
    print(f"  {'-'*W}")

    screen_results = []
    for sym in sorted(valid):
        r = screen_stock(sym, all_data[sym], index_bh_net)
        if r is None: continue
        if r["ta_return"] > 0 and r["trades"] >= 3:    cls = "TA-RESPONSIVE"
        elif r["trades"] == 0 and r["bh_return"] > 30: cls = "MOMENTUM"
        elif r["ta_return"] < -10:                      cls = "AVOID"
        else:                                            cls = "WEAK"
        r["class"] = cls
        screen_results.append(r)
        print(f"  {r['symbol']:<8}  {r['ta_return']:>+7.1f}%  {r['bh_return']:>+8.1f}%  "
              f"  {r['alpha_idx']:>+9.1f}%  {r['trades']:>7}  {r['win_rate']:>5.0f}%  {cls:>16}")

    screen_results.sort(key=lambda x: x["alpha_idx"], reverse=True)
    ta_cls  = [r["symbol"] for r in screen_results if r["class"] == "TA-RESPONSIVE"]
    mom_cls = [r["symbol"] for r in screen_results if r["class"] == "MOMENTUM"]
    avoid   = [r["symbol"] for r in screen_results if r["class"] == "AVOID"]
    beats   = [r["symbol"] for r in screen_results if r["alpha_idx"] > 0]

    print(f"\n  TA-RESPONSIVE ({len(ta_cls)}): {', '.join(ta_cls)}")
    print(f"  MOMENTUM      ({len(mom_cls)}): {', '.join(mom_cls)}")
    print(f"  AVOID         ({len(avoid)}): {', '.join(avoid)}")
    print(f"  Beats index individually ({len(beats)}): {', '.join(beats) or 'none'}")

    # ══ PHASE 2: Portfolio backtest ══════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("  PHASE 2 — PORTFOLIO BACKTEST  |  v9-13 Strategy (No-TP + Risk-Parity)")
    print(f"  $10,000 starting capital  |  25% CGT  |  US Treasury bonds in NEUTRAL/BEAR")
    print("=" * W)

    print("\n  Running portfolio backtest...")
    r = run_backtest(valid, all_data, index_df)
    if r is None:
        print("  No results."); return

    # Regime breakdown
    total_d = sum(r["regime_days"].values())
    print(f"\n  REGIME BREAKDOWN  ({total_d} trading days)")
    print(f"  {'-'*W}")
    for rg, d in r["regime_days"].items():
        bar = "#" * int(d/total_d*60)
        print(f"  {rg:<8} {d:>4}d ({d/total_d*100:>4.1f}%)  {bar}")

    print(f"\n  REGIME TIMELINE")
    for ds, rg in r["regime_log"][:20]:
        labels = {"BULL":"trend-following + no TP",
                  "NEUTRAL":"mean-reversion + 40% bonds",
                  "BEAR":"full exit to 60% bonds"}
        print(f"    {ds}  {rg:<8}  [{labels[rg]}]")
    if len(r["regime_log"]) > 20:
        print(f"    ... ({len(r['regime_log'])-20} more transitions)")

    # P&L by stock
    all_syms = sorted({t.symbol for t in [] if t.side=="SELL"})  # placeholder
    sell_trades = [t for t in [] if t.side=="SELL"]

    # Open positions
    print(f"\n  TOP OPEN POSITIONS (as of 2026-03-31):")
    print(f"  {'Symbol':<8}  {'Qty':>8}  {'AvgCost':>10}  {'Last':>10}  "
          f"{'MktVal':>10}  {'Unrealised':>12}  {'Ret%':>8}")
    print(f"  {'-'*80}")
    for sym, qty, avg, last, mv, up, upct in r["open_pos"][:10]:
        print(f"  {sym:<8}  {qty:>8.2f}  ${avg:>9,.2f}  ${last:>9,.2f}  "
              f"${mv:>9,.2f}  ${up:>+11,.2f}  {upct:>+7.1f}%")
    if len(r["open_pos"]) > 10:
        print(f"  ... ({len(r['open_pos'])-10} more open positions)")

    # ══ FINAL COMPARISON ════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print(f"  FINAL RESULTS  |  NASDAQ 100  |  $10,000  |  April 2021 - March 2026")
    print(f"  {'-'*W}")

    print(f"\n  STRATEGY (v9-13: No-TP in BULL + Risk-Parity sizing + Bond rotation)")
    print(f"  {'─'*60}")
    print(f"  Starting Capital:              ${INITIAL_CASH:>10,.2f}")
    print(f"  Cash on Hand:                  ${r['cash']:>10,.2f}")
    print(f"  Open Stock Positions (Mkt):    ${r['stock_mkt']:>10,.2f}")
    print(f"  Open Bond Holdings:            ${r['bond_mkt']:>10,.2f}")
    print(f"  TOTAL PORTFOLIO VALUE:         ${r['total_val']:>10,.2f}")
    print(f"  {'─'*60}")
    print(f"  Total Return (net of 25% tax):     {r['total_ret']:>+8.2f}%")
    print(f"  CAGR (5yr):                        {r['cagr']:>+8.2f}%")
    print(f"  Avg Capital Utilization:           {r['avg_util']:>7.1f}%")
    print(f"  Stock Win Rate:                    {r['win_rate']:>7.0f}%")
    print(f"  Avg Win / Loss:                ${r['avg_win']:>+7,.2f}  /  ${r['avg_loss']:>+7,.2f}")
    print(f"  R/R Ratio:                         {r['rr']:>7.2f}x")
    print(f"  Exits:  TP={r['tp_exits']}  Stop={r['stop_exits']}  "
          f"Bear={r['bear_exits']}  Signal={r['sig_exits']}")
    print(f"  Total Tax Paid:                ${r['total_tax']:>10,.2f}")
    print(f"  Bond Net P&L:                  ${r['bond_net']:>+9,.2f}")

    print(f"\n  {'═'*W}")
    print(f"  SIDE-BY-SIDE  |  Strategy vs NASDAQ 100 Buy & Hold  |  $10,000")
    print(f"  {'─'*W}")
    print(f"  {'Metric':<40}  {'Strategy (v9-13)':>20}  {'NASDAQ B&H':>18}")
    print(f"  {'─'*W}")

    bh_val  = INITIAL_CASH * (1 + bm_gross/100)
    bh_tax  = max(0, (bh_val - INITIAL_CASH) * TAX_RATE)
    bh_net  = bh_val - bh_tax

    rows = [
        ("Final Portfolio Value",     f"${r['total_val']:>10,.2f}",    f"${bh_net:>10,.2f}"),
        ("Total Return (net)",        f"{r['total_ret']:>+9.2f}%",     f"{index_bh_net:>+9.2f}%"),
        ("CAGR (5yr)",                f"{r['cagr']:>+9.2f}%",          f"{((1+index_bh_net/100)**(1/5)-1)*100:>+9.2f}%"),
        ("Capital Utilization",       f"{r['avg_util']:>9.1f}%",       f"{'100':>9}%"),
        ("Tax Paid",                  f"${r['total_tax']:>10,.2f}",     f"${bh_tax:>10,.2f}"),
        ("Number of Trades",          f"{r['n_buys']:>10} buys",        "1 buy, 1 sell"),
        ("Gross Index Return",        f"(same period)",                 f"{bm_gross:>+9.1f}%"),
    ]
    for name, strat, bh in rows:
        print(f"  {name:<40}  {strat:>20}  {bh:>18}")

    # Alpha
    alpha = r['total_ret'] - index_bh_net
    alpha_mark = "BEATS B&H" if alpha > 0 else "below B&H"
    print(f"  {'─'*W}")
    print(f"  {'Alpha vs NASDAQ 100 B&H':<40}  {alpha:>+19.2f}%  ({alpha_mark})")
    print(f"  {'═'*W}\n")

    # TA-125 cross-market comparison
    print(f"  CROSS-MARKET COMPARISON  |  Same strategy, same period, same $10k")
    print(f"  {'─'*W}")
    print(f"  {'Market':<20}  {'Return':>10}  {'CAGR':>8}  {'B&H Net':>10}  {'Alpha':>9}")
    print(f"  {'─'*W}")
    ta125_net = 105.13; ta125_bh_cagr = 15.5
    ndx_cagr  = ((1+index_bh_net/100)**(1/5)-1)*100
    strat_cagr_ta125 = ((1+131.07/100)**(1/5)-1)*100
    print(f"  {'TA-125 (Israel)':<20}  {'+131.07%':>10}  {strat_cagr_ta125:>+7.1f}%  "
          f"{ta125_net:>+9.1f}%  {'+131.07-105.13=+25.94':>9}")
    print(f"  {'NASDAQ 100 (US)':<20}  {r['total_ret']:>+9.2f}%  {r['cagr']:>+7.2f}%  "
          f"{index_bh_net:>+9.1f}%  {alpha:>+8.2f}%")
    print(f"  {'─'*W}")
    print(f"  NOTE: TA-125 result uses in-sample optimization (same 5yr period).")
    print(f"        NASDAQ is an out-of-sample test of the same strategy.")
    print(f"  {'═'*W}\n")


if __name__ == "__main__":
    run()
