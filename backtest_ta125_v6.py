"""
TA-125 Technical Analysis Backtest — v6: Alpha-Stock Screener + Strategy
Period: April 2021 - March 2026 (5 years)

Phase 1 — Stock Screener:
  For every stock in the TA-125 universe, run an independent 5-year TA backtest
  (signal-driven: buy on BULLISH, sell on BEARISH / stop / take-profit).
  Keep only stocks where TA-return > TA-125 index return (net of tax).

Phase 2 — Portfolio Backtest:
  Run the v4-style adaptive-regime strategy using ONLY the screened universe.
  + Bond rotation during BEAR (60% bonds) from v5.

NOTE: Stock selection is done in-sample (same 5-year period).
      This identifies TA-responsive stocks for forward use.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass
import yfinance as yf
import pandas as pd
import ta

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
INITIAL_CASH   = 100_000.0
COMMISSION     = 0.0008
TAX_RATE       = 0.25
RS_LOOKBACK    = 63
ETF_COMMISSION = 0.0005

INDEX_TICKER   = "^TA125.TA"
SP500_ETF      = "SPEN.TA"

SIM_START   = pd.Timestamp("2021-04-01")
SIM_END     = pd.Timestamp("2026-03-31")
FETCH_START = "2020-01-01"
FETCH_END   = "2026-04-01"

BOND_ANNUAL_YIELD = {2021: 0.015, 2022: 0.010, 2023: 0.043, 2024: 0.040, 2025: 0.038, 2026: 0.038}

# Screening params (single-stock backtest)
SCREEN_MIN_BULL   = 4      # min bull signals to buy
SCREEN_MIN_BEAR   = 4      # min bear signals to sell
SCREEN_STOP_PCT   = 0.08   # 8% hard stop
SCREEN_TP_PCT     = 0.35   # 35% take-profit
SCREEN_ATR_MULT   = 3.0    # ATR trailing stop multiplier
SCREEN_MIN_HOLD   = 5      # days before signal-based exit

# Portfolio strategy params (v4-style)
REGIME_PARAMS = {
    "BULL":    (3.5, 0.08, 0.35, 4, 4, 10, 0.12, 15, 3.0, False),
    "NEUTRAL": (2.0, 0.06, 0.12, 4, 3,  5, 0.08,  5, 0.0,  True),
    "BEAR":    (1.0, 0.05, 0.10, 9, 2,  0, 0.05,  0, 0.0, False),
}
REGIME_ETF_ALLOC = {
    "BULL":    (0.00, 0.00),
    "NEUTRAL": (0.00, 0.40),   # bonds only in NEUTRAL (no SPEN)
    "BEAR":    (0.00, 0.60),   # bonds only in BEAR
}
MIN_CASH_RESERVE_PCT = 0.05

TA125_UNIVERSE = [
    "POLI.TA","LUMI.TA","DSCT.TA","MZRH.TA","FIBI.TA",
    "NICE.TA","CAMT.TA","TSEM.TA","NVMI.TA","SPNS.TA","ITRN.TA",
    "ESLT.TA","TEVA.TA","ICL.TA","BEZQ.TA",
    "SKBN.TA","RSEL.TA",
    "PHNX.TA","HARL.TA","MGDL.TA","MNRN.TA",
    "AZRG.TA","AMOT.TA","ALHE.TA","ELCO.TA",
    "ENLT.TA","DLEKG.TA","BAZAN.TA","ILCO.TA",
]


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
    note: str = ""; signals_bull: int = 0; signals_bear: int = 0
    regime: str = ""; category: str = "STOCK"


# ─── Shared helpers ───────────────────────────────────────────────────────────
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
    c = float(close.iloc[-1])
    if sma200 is None: return "BULL"
    sna = sma50_s.dropna()
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

def bond_daily_return(day: pd.Timestamp) -> float:
    return BOND_ANNUAL_YIELD.get(day.year, 0.040) / 252

def compute_signals(df_slice, min_bull, min_bear):
    if len(df_slice) < 30:
        return {"bullish":0,"bearish":0,"bias":"INSUFFICIENT","rsi":None,
                "bull_details":[],"bear_details":[],"bb_pct":None,"close":None}
    close=df_slice["Close"]; high=df_slice["High"]; low=df_slice["Low"]
    sma20=ta.trend.sma_indicator(close,20); sma50=ta.trend.sma_indicator(close,50)
    sma200=ta.trend.sma_indicator(close,200) if len(df_slice)>=200 \
           else pd.Series([float("nan")]*len(df_slice),index=close.index)
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


# ─── Phase 1: Per-stock screening backtest ────────────────────────────────────
def screen_single_stock(sym: str, df: pd.DataFrame, index_bh_net: float) -> dict:
    """
    Run a simple signal-driven backtest on a single stock.
    Returns screening result dict.
    """
    sim_days = [d for d in df.index if SIM_START <= d <= SIM_END]
    if len(sim_days) < 60:
        return None

    capital = 10_000.0
    cash    = capital
    pos     = None   # dict with qty/avg_cost/buy_comm/trail_high/entry_idx
    trades  = []

    for day_idx, day in enumerate(sim_days):
        df_sl = df[df.index <= day].tail(260)
        price = float(df.loc[day, "Close"])

        # Update trail high
        if pos and price > pos["trail_high"]:
            pos["trail_high"] = price

        # Exit logic
        if pos:
            chg       = (price - pos["avg_cost"]) / pos["avg_cost"]
            hold_days = day_idx - pos["entry_idx"]
            atr       = compute_atr(df_sl)
            floor     = pos["avg_cost"] * (1 - SCREEN_STOP_PCT)
            trail_stop = pos["trail_high"] - atr * SCREEN_ATR_MULT if atr else floor
            eff_stop  = max(trail_stop, floor)

            sell_reason = None
            if price <= eff_stop:
                sell_reason = f"STOP ({chg*100:+.1f}%)"
            elif hold_days >= SCREEN_MIN_HOLD:
                sig = compute_signals(df_sl, SCREEN_MIN_BULL, SCREEN_MIN_BEAR)
                if chg >= SCREEN_TP_PCT and sig["bias"] != "BULLISH":
                    sell_reason = f"TP ({chg*100:+.1f}%)"
                elif sig["bias"] == "BEARISH" and sig["bearish"] >= SCREEN_MIN_BEAR:
                    sell_reason = f"SIGNAL ({chg*100:+.1f}%)"

            if sell_reason:
                sc  = pos["qty"] * price * COMMISSION
                gp  = (price - pos["avg_cost"]) * pos["qty"]
                tx  = gp - pos["buy_comm"] - sc
                tax = max(0.0, tx * TAX_RATE)
                cash += pos["qty"] * price - sc - tax
                trades.append({"gross": gp, "net": tx - tax, "reason": sell_reason})
                pos = None

        # Entry logic
        if pos is None and cash >= 500:
            sig = compute_signals(df_sl, SCREEN_MIN_BULL, SCREEN_MIN_BEAR)
            if sig["bias"] == "BULLISH" and sig["bullish"] >= SCREEN_MIN_BULL:
                bp  = price
                qty = int(cash * 0.95 / (bp * (1 + COMMISSION)))
                if qty >= 1:
                    bc   = qty * bp * COMMISSION
                    cash -= qty * bp + bc
                    pos  = {"qty": qty, "avg_cost": bp, "buy_comm": bc,
                            "trail_high": bp, "entry_idx": day_idx}

    # Close open position at end
    if pos and sim_days:
        last_day = sim_days[-1]
        price = float(df.loc[last_day, "Close"])
        sc  = pos["qty"] * price * COMMISSION
        gp  = (price - pos["avg_cost"]) * pos["qty"]
        tx  = gp - pos["buy_comm"] - sc
        tax = max(0.0, tx * TAX_RATE)
        cash += pos["qty"] * price - sc - tax
        trades.append({"gross": gp, "net": tx - tax, "reason": "OPEN_CLOSE"})

    ta_return = (cash - capital) / capital * 100

    # Buy-and-hold for this stock (net of tax)
    hist = df[(df.index >= SIM_START) & (df.index <= SIM_END)]
    bh_return = 0.0
    if len(hist) >= 2:
        p0, p1 = float(hist["Close"].iloc[0]), float(hist["Close"].iloc[-1])
        bh_gain = capital * (p1/p0 - 1)
        bh_return = (bh_gain - max(0, bh_gain * TAX_RATE)) / capital * 100

    wins   = [t for t in trades if t["net"] > 0]
    losses = [t for t in trades if t["net"] <= 0]
    wr     = len(wins)/len(trades)*100 if trades else 0.0
    alpha_vs_index = ta_return - index_bh_net
    alpha_vs_stock = ta_return - bh_return

    return {
        "symbol":    sym,
        "ta_return": round(ta_return, 2),
        "bh_return": round(bh_return, 2),
        "alpha_idx": round(alpha_vs_index, 2),
        "alpha_stk": round(alpha_vs_stock, 2),
        "trades":    len(trades),
        "win_rate":  round(wr, 1),
        "avg_win":   round(sum(t["net"] for t in wins)/len(wins), 0) if wins else 0,
        "avg_loss":  round(sum(t["net"] for t in losses)/len(losses), 0) if losses else 0,
        "beats_index": ta_return > index_bh_net,
    }


# ─── Phase 2: Portfolio backtest (v4-style + bond rotation) ───────────────────
def run_portfolio_backtest(valid_stocks: list, all_data: dict, index_df,
                           universe_label: str):
    """Run the adaptive-regime portfolio backtest on a given stock universe."""
    all_dates = set()
    for df in all_data.values(): all_dates.update(df.index.tolist())
    if index_df is not None: all_dates.update(index_df.index.tolist())
    trading_days = sorted(d for d in all_dates if SIM_START <= d <= SIM_END)
    if not trading_days: return None

    cash        = INITIAL_CASH
    positions:  dict[str, Position]   = {}
    etf_holdings: dict[str, EtfHolding] = {}
    trades:     list[Trade]            = []
    total_tax   = 0.0
    last_regime = None
    regime_days = {"BULL": 0, "NEUTRAL": 0, "BEAR": 0}
    daily_utilization = []

    for day_idx, day in enumerate(trading_days):
        day_str = day.strftime("%Y-%m-%d")
        regime  = classify_regime(index_df, day) if index_df is not None else "BULL"
        (atr_mult, init_stop, tp_pct, min_bull_buy, min_bear_exit,
         max_pos, pos_pct, min_hold, rs_min, mean_rev) = REGIME_PARAMS[regime]
        _, bond_alloc = REGIME_ETF_ALLOC[regime]

        regime_days[regime] += 1
        if regime != last_regime:
            last_regime = regime

        # Update trail highs
        for sym, pos in positions.items():
            if sym in all_data and day in all_data[sym].index:
                cp = float(all_data[sym].loc[day, "Close"])
                if cp > pos.trail_high: pos.trail_high = cp

        # Accrue bond interest
        if "BOND" in etf_holdings:
            etf_holdings["BOND"].quantity *= (1 + bond_daily_return(day))

        # ── STOCK SELLS ─────────────────────────────────────────────────────
        to_sell = []
        for sym, pos in list(positions.items()):
            if sym not in all_data or day not in all_data[sym].index: continue
            cp  = float(all_data[sym].loc[day, "Close"])
            chg = (cp - pos.avg_cost) / pos.avg_cost
            hold_days = day_idx - pos.entry_day_idx
            e_atr_mult, e_init_stop, e_tp_pct, _, e_min_bear, _, _, e_min_hold, _, _ = \
                REGIME_PARAMS[pos.regime_at_buy]
            df_sl = all_data[sym][all_data[sym].index <= day].tail(260)
            sig   = compute_signals(df_sl, 3, e_min_bear)
            atr   = compute_atr(df_sl)
            floor      = pos.avg_cost * (1 - e_init_stop)
            trail_stop = pos.trail_high - atr * e_atr_mult if atr else floor
            eff_stop   = max(trail_stop, floor)
            if cp <= eff_stop:
                pct = chg*100; peak = (pos.trail_high/pos.avg_cost - 1)*100
                atr_s = f"{atr:.1f}" if atr else "n/a"
                to_sell.append((sym, f"TRAIL_STOP ({pct:+.1f}%, peak+{peak:.1f}%, ATR={atr_s})", sig))
                continue
            if regime == "BEAR":
                to_sell.append((sym, "BEAR_EXIT", sig)); continue
            if chg >= e_tp_pct and hold_days >= e_min_hold and sig["bias"] in ("BEARISH","NEUTRAL"):
                to_sell.append((sym, f"TAKE_PROFIT ({chg*100:.1f}%, held {hold_days}d)", sig)); continue
            if hold_days >= e_min_hold and sig["bias"] == "BEARISH" and sig["bearish"] >= e_min_bear:
                to_sell.append((sym, f"BEARISH ({sig['bearish']}b/{sig['bullish']}B, held {hold_days}d)", sig))

        for sym, reason, sig in to_sell:
            pos = positions[sym]
            sp  = float(all_data[sym].loc[day, "Close"])
            sc  = pos.quantity * sp * COMMISSION
            gp  = (sp - pos.avg_cost) * pos.quantity
            tx  = gp - pos.buy_commission - sc
            tax = max(0.0, tx * TAX_RATE)
            net = tx - tax
            cash += pos.quantity * sp - sc - tax
            total_tax += tax
            trades.append(Trade(date=day_str, symbol=sym, side="SELL",
                quantity=pos.quantity, price=round(sp,3), commission=round(sc,2),
                gross_pnl=round(gp,2), taxable_pnl=round(tx,2),
                tax=round(tax,2), net_pnl=round(net,2),
                note=reason, signals_bull=sig.get("bullish",0),
                signals_bear=sig.get("bearish",0), regime=regime, category="STOCK"))
            del positions[sym]

        # ── BOND SELL on regime change to 0 alloc ───────────────────────────
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
                taxable_pnl=round(tx,2), tax=round(tax,2),
                net_pnl=round(tx-tax,2),
                note=f"BOND_LIQUIDATE (regime={regime})", regime=regime, category="BOND"))
            del etf_holdings["BOND"]

        # ── STOCK BUYS ──────────────────────────────────────────────────────
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
                tag = "MR" if mean_rev else f"RS+{rs:.1f}%"
                trades.append(Trade(date=day_str, symbol=sym, side="BUY",
                    quantity=qty, price=round(bp,3), commission=round(bc,2),
                    note=f"[{regime}|{tag}|{sig['bullish']}B] " + "; ".join(sig["bull_details"][:2]),
                    signals_bull=sig["bullish"], signals_bear=sig["bearish"],
                    regime=regime, category="STOCK"))

        # ── BOND BUY — deploy idle cash ─────────────────────────────────────
        pos_val = sum(
            p.quantity * float(all_data[p.symbol].loc[day,"Close"])
            for p in positions.values()
            if p.symbol in all_data and day in all_data[p.symbol].index
        )
        bond_val = etf_holdings["BOND"].quantity if "BOND" in etf_holdings else 0.0
        portfolio_val = cash + pos_val + bond_val
        reserve    = portfolio_val * MIN_CASH_RESERVE_PCT
        deployable = max(0.0, cash - reserve)
        target_bond = portfolio_val * bond_alloc

        if bond_alloc > 0 and deployable > 500:
            if bond_val < target_bond * 0.90:
                to_invest = min(target_bond - bond_val, deployable * 0.9)
                bc   = to_invest * ETF_COMMISSION
                cost = to_invest + bc
                if cost <= cash:
                    cash -= cost
                    if "BOND" in etf_holdings:
                        h = etf_holdings["BOND"]
                        h.avg_cost += to_invest
                        h.quantity += to_invest
                        h.buy_commission += bc
                    else:
                        etf_holdings["BOND"] = EtfHolding("BOND", to_invest, to_invest, bc)
                    trades.append(Trade(date=day_str, symbol="BOND", side="BUY",
                        quantity=round(to_invest,2), price=1.0, commission=round(bc,2),
                        note=f"[{regime}] Bond rotation (~{BOND_ANNUAL_YIELD.get(day.year,0.04)*100:.1f}%/yr)",
                        regime=regime, category="BOND"))

        # Utilization
        total_invested = pos_val + bond_val
        total_port = cash + total_invested
        daily_utilization.append(total_invested / total_port * 100 if total_port > 0 else 0)

    # ── Final valuation ───────────────────────────────────────────────────────
    last_day = trading_days[-1]
    stock_mkt = 0.0
    open_stocks = []
    for sym, pos in positions.items():
        df = all_data.get(sym)
        lp = float(df.loc[last_day,"Close"]) if df is not None and last_day in df.index else pos.avg_cost
        mv = pos.quantity * lp
        up = mv - (pos.quantity * pos.avg_cost + pos.buy_commission)
        stock_mkt += mv
        open_stocks.append((sym, pos, lp, mv, up, (lp/pos.avg_cost-1)*100))

    bond_mkt = etf_holdings["BOND"].quantity if "BOND" in etf_holdings else 0.0
    total_val = cash + stock_mkt + bond_mkt
    total_ret = (total_val - INITIAL_CASH) / INITIAL_CASH * 100
    avg_util  = sum(daily_utilization) / len(daily_utilization) if daily_utilization else 0

    sell_stock = [t for t in trades if t.side=="SELL" and t.category=="STOCK"]
    wins   = [t for t in sell_stock if t.net_pnl > 0]
    losses = [t for t in sell_stock if t.net_pnl <= 0]
    wr     = len(wins)/len(sell_stock)*100 if sell_stock else 0
    avg_win  = sum(t.net_pnl for t in wins)/len(wins)   if wins   else 0
    avg_loss = sum(t.net_pnl for t in losses)/len(losses) if losses else 0

    return {
        "label":      universe_label,
        "total_val":  total_val,
        "total_ret":  total_ret,
        "cagr":       ((1+total_ret/100)**(1/5)-1)*100,
        "avg_util":   avg_util,
        "win_rate":   wr,
        "avg_win":    avg_win,
        "avg_loss":   avg_loss,
        "stock_buys": len([t for t in trades if t.side=="BUY" and t.category=="STOCK"]),
        "stock_sells":len(sell_stock),
        "bond_net":   sum(t.net_pnl for t in trades if t.side=="SELL" and t.category=="BOND"),
        "total_tax":  total_tax,
        "open_stocks": open_stocks,
        "cash":       cash,
        "stock_mkt":  stock_mkt,
        "bond_mkt":   bond_mkt,
        "trades":     trades,
        "regime_days": regime_days,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def run():
    print("Fetching data...")

    # Index
    index_df = None
    try:
        idx = yf.Ticker(INDEX_TICKER).history(start=FETCH_START, end=FETCH_END)
        if not idx.empty:
            idx.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in idx.index])
            index_df = idx
            print(f"  Index {INDEX_TICKER}: {len(idx)} bars")
    except Exception as e:
        print(f"  Index fetch failed: {e}")

    # Index benchmark (net of tax)
    index_bh_net = 0.0
    if index_df is not None:
        sim_idx = index_df[(index_df.index >= SIM_START) & (index_df.index <= SIM_END)]
        if len(sim_idx) >= 2:
            p0, p1 = float(sim_idx["Close"].iloc[0]), float(sim_idx["Close"].iloc[-1])
            bm_gross = (p1/p0 - 1)*100
            gain = INITIAL_CASH*(p1/p0 - 1)
            index_bh_net = (gain - max(0, gain*TAX_RATE)) / INITIAL_CASH * 100
            print(f"  Index B&H: gross {bm_gross:+.1f}%, net {index_bh_net:+.1f}%")

    # Stocks
    all_data: dict[str, pd.DataFrame] = {}
    for sym in TA125_UNIVERSE:
        try:
            df = yf.Ticker(sym).history(start=FETCH_START, end=FETCH_END)
            if df.empty or len(df) < 50: continue
            df.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in df.index])
            all_data[sym] = df
        except Exception: pass
    print(f"  Stocks loaded: {len(all_data)}\n")

    W = 120

    # ══════════════════════════════════════════════════════════════════════════
    print("=" * W)
    print("  PHASE 1 — PER-STOCK TA SCREENING  |  April 2021 - March 2026")
    print(f"  Buy: {SCREEN_MIN_BULL}+ bull signals | Sell: {SCREEN_MIN_BEAR}+ bear / ATR trail / TP {SCREEN_TP_PCT*100:.0f}%")
    print("=" * W)
    print(f"\n  {'Symbol':<10}  {'TA%':>8}  {'B&H%':>8}  {'AlphaIdx':>9}  {'AlphaStk':>9}  "
          f"{'Trades':>7}  {'Win%':>6}  {'AvgWin':>8}  {'AvgLoss':>9}  {'Beats?':>7}")
    print(f"  {'-'*W}")

    screen_results = []
    for sym in sorted(all_data.keys()):
        r = screen_single_stock(sym, all_data[sym], index_bh_net)
        if r is None: continue
        screen_results.append(r)
        beat = "YES ***" if r["beats_index"] else "no"
        print(f"  {r['symbol']:<10}  {r['ta_return']:>+7.1f}%  {r['bh_return']:>+7.1f}%  "
              f"  {r['alpha_idx']:>+8.1f}%  {r['alpha_stk']:>+8.1f}%  "
              f"{r['trades']:>7}  {r['win_rate']:>5.0f}%  "
              f"NIS{r['avg_win']:>+6,.0f}  NIS{r['avg_loss']:>+7,.0f}  {beat:>7}")

    # Sort by alpha vs index
    screen_results.sort(key=lambda x: x["alpha_idx"], reverse=True)
    alpha_stocks = [r["symbol"] for r in screen_results if r["beats_index"]]
    losing_stocks = [r["symbol"] for r in screen_results if not r["beats_index"]]

    print(f"\n  {'─'*W}")
    print(f"  Stocks that beat the index with TA signals ({len(alpha_stocks)}): "
          f"{', '.join(alpha_stocks)}")
    print(f"  Stocks that did NOT beat the index ({len(losing_stocks)}): "
          f"{', '.join(losing_stocks)}")

    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("  PHASE 2 — PORTFOLIO BACKTEST  |  Full universe vs Alpha-filtered universe")
    print("=" * W)

    # Full universe
    full_valid = [s for s in all_data.keys()]
    print(f"\n  Running full-universe portfolio ({len(full_valid)} stocks)...")
    full_result = run_portfolio_backtest(full_valid, all_data, index_df,
                                         f"Full universe ({len(full_valid)} stocks)")

    # Alpha-filtered universe
    alpha_valid = [s for s in alpha_stocks if s in all_data]
    print(f"  Running alpha-filtered portfolio ({len(alpha_valid)} stocks)...")
    alpha_result = run_portfolio_backtest(alpha_valid, all_data, index_df,
                                          f"Alpha stocks only ({len(alpha_valid)} stocks)")

    # ── Print results ─────────────────────────────────────────────────────────
    for res in [full_result, alpha_result]:
        if res is None: continue
        print(f"\n  {'─'*W}")
        print(f"  {res['label'].upper()}")
        print(f"  {'─'*W}")
        print(f"  Total Portfolio Value:   NIS {res['total_val']:>12,.2f}")
        print(f"  Total Return (5yr net):      {res['total_ret']:>+10.2f}%")
        print(f"  CAGR:                        {res['cagr']:>+10.2f}%")
        print(f"  Avg Capital Utilization:     {res['avg_util']:>9.1f}%")
        print(f"  Stock Win Rate:              {res['win_rate']:>9.0f}%")
        print(f"  Avg Win / Loss:          NIS {res['avg_win']:>+7,.0f} / NIS {res['avg_loss']:>+7,.0f}")
        print(f"  R/R Ratio:                   {abs(res['avg_win']/res['avg_loss']) if res['avg_loss'] else 0:>9.2f}x")
        print(f"  Stock Trades (B/S):          {res['stock_buys']:>4} / {res['stock_sells']:<4}")
        print(f"  Bond rotation net P&L:   NIS {res['bond_net']:>+10,.2f}")
        print(f"  Total Tax Paid:          NIS {res['total_tax']:>12,.2f}")

        if res["open_stocks"]:
            print(f"\n  Open Positions (2026-03-31):")
            for sym, pos, lp, mv, up, upct in res["open_stocks"]:
                print(f"    {sym:<10}  qty={pos.quantity}  avg={pos.avg_cost:,.0f}  "
                      f"last={lp:,.0f}  MV={mv:,.0f}  PnL={up:+,.0f} ({upct:+.1f}%)")

    # ── Final comparison table ─────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  5-YEAR COMPARISON  |  April 2021 - March 2026  |  NIS 100,000")
    print(f"  {'-'*W}")
    print(f"  {'Strategy':<62}  {'Return':>9}  {'CAGR':>7}  {'Util':>6}  {'WinRate':>8}")
    print(f"  {'-'*W}")
    rows = [
        ("v4  Adaptive Regime (stocks only)",                        66.16,  10.7,  33, 60),
        ("v5  + Bond rotation (BEAR/NEUTRAL, SPEN removed)",         67.37,  10.85, 67, 59),
    ]
    if full_result:
        rows.append((f"v6a Full universe + bonds ({len(full_valid)} stocks)",
                     full_result["total_ret"], full_result["cagr"],
                     full_result["avg_util"],  full_result["win_rate"]))
    if alpha_result:
        rows.append((f"v6b Alpha stocks + bonds ({len(alpha_valid)} stocks, beat index)",
                     alpha_result["total_ret"], alpha_result["cagr"],
                     alpha_result["avg_util"],  alpha_result["win_rate"]))
    rows.append(("    Buy & Hold TA-125 (gross +140.2%, net after tax)",
                 105.13, 15.5, 100, 0))

    for name, ret, cagr, ut, wr in rows:
        wr_s = f"{wr:.0f}%" if wr else "  -"
        print(f"  {name:<62}  {ret:>+8.2f}%  {cagr:>+6.1f}%  {ut:>5.0f}%  {wr_s:>7}")
    print(f"  {'=' * W}\n")


if __name__ == "__main__":
    run()
