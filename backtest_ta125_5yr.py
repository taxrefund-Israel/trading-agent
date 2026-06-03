"""
TA-125 Technical Analysis Backtest — Index-Beating Strategy (v4)
Period: April 2021 - March 2026 (TASE trading days)

Root-cause analysis from v3:
  - Buy & Hold beat us by +31% despite worse tax efficiency
  - We were only 40-60% deployed on average (too much idle cash)
  - Too many exits paid 25% tax, reducing compounding capital
  - Sold winners too early (ENLT sold at +15% then ran to +83%)

v4 design principles:
  1. Always deploy — 10 concentrated positions × 12% = fully invested in BULL
  2. Hold winners hard — ATR 3.5x stop, TP only at 35%, min 15-day hold
  3. Quality-only entry — RS >3% AND score ≥4 bullish signals
  4. Bear-proof — exit ALL positions within 2 days of BEAR detection; sit in cash
  5. NEUTRAL — tighten stops to 2×ATR on existing, no new buys except deep dips
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass
import yfinance as yf
import pandas as pd
import ta

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────────
INITIAL_CASH   = 100_000.0
COMMISSION     = 0.0008
TAX_RATE       = 0.25
RS_LOOKBACK    = 63

INDEX_TICKER = "^TA125.TA"
SIM_START    = pd.Timestamp("2021-04-01")
SIM_END      = pd.Timestamp("2026-03-31")
FETCH_START  = "2020-01-01"
FETCH_END    = "2026-04-01"

# Regime-adaptive params
# (atr_mult, init_stop, tp_pct, min_bull_buy, min_bear_exit, max_pos, pos_pct, min_hold_days, rs_min, mean_rev)
REGIME_PARAMS = {
    "BULL":    (3.5, 0.08, 0.35, 4, 4, 10, 0.12, 15, 3.0, False),
    "NEUTRAL": (2.0, 0.06, 0.12, 4, 3,  5, 0.08,  5, 0.0,  True),
    "BEAR":    (1.0, 0.05, 0.10, 9, 2,  0, 0.05,  0, 0.0, False),  # max_pos=0 → no new buys
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


# ─── Data classes ─────────────────────────────────────────────────────────────────
@dataclass
class Position:
    symbol:         str
    quantity:       int
    avg_cost:       float
    buy_commission: float
    trail_high:     float
    regime_at_buy:  str
    entry_day_idx:  int       # index into trading_days list for hold-period check

@dataclass
class Trade:
    date: str; symbol: str; side: str; quantity: int; price: float
    commission: float; gross_pnl: float = 0.0; taxable_pnl: float = 0.0
    tax: float = 0.0; net_pnl: float = 0.0
    note: str = ""; signals_bull: int = 0; signals_bear: int = 0; regime: str = ""


# ─── Helpers ──────────────────────────────────────────────────────────────────────
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
    above = c > sma200
    if above and slope > 0.008 and (adx or 0) > 22: return "BULL"
    if (not above) and slope < -0.008:               return "BEAR"
    return "NEUTRAL"

def relative_strength(stock_df, index_df, day, lookback=RS_LOOKBACK):
    sh = stock_df[stock_df.index <= day]["Close"]
    ih = index_df[index_df.index <= day]["Close"]
    if len(sh) < lookback + 1 or len(ih) < lookback + 1: return None
    return round((float(sh.iloc[-1])/float(sh.iloc[-lookback]) - 1)*100 -
                 (float(ih.iloc[-1])/float(ih.iloc[-lookback]) - 1)*100, 2)

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
    if vs20: (bull if c>vs20 else bear).append(f"{'above' if c>vs20 else 'below'} SMA20")
    if vs50: (bull if c>vs50 else bear).append(f"{'above' if c>vs50 else 'below'} SMA50")
    if vs200 and not pd.isna(vs200): (bull if c>vs200 else bear).append(f"{'above' if c>vs200 else 'below'} SMA200")
    if ve12 and ve26: (bull if ve12>ve26 else bear).append(f"EMA12 {'>' if ve12>ve26 else '<'} EMA26")
    if bb_pct is not None:
        if c<vbbl:       bull.append("Below lower BB")
        elif c>vbbu:     bear.append("Above upper BB")
        elif bb_pct<0.3: bull.append(f"Lower BB ({bb_pct*100:.0f}%)")
        elif bb_pct>0.7: bear.append(f"Upper BB ({bb_pct*100:.0f}%)")
    if vsk is not None and vsd is not None:
        if vsk<25 and vsd<25: bull.append(f"Stoch oversold ({vsk:.0f})")
        elif vsk>75 and vsd>75: bear.append(f"Stoch overbought ({vsk:.0f})")
    nb,nb2=len(bull),len(bear)
    if nb>=min_bull and nb>nb2+1:   bias="BULLISH"
    elif nb2>=min_bear and nb2>nb+1: bias="BEARISH"
    else:                             bias="NEUTRAL"
    return {"bullish":nb,"bearish":nb2,"bias":bias,
            "rsi":round(vrsi,1) if vrsi else None,
            "close":c,"bb_pct":bb_pct,"bull_details":bull,"bear_details":bear}

def is_mean_reversion_entry(sig):
    rsi=sig.get("rsi") or 999
    bp=sig.get("bb_pct")
    return bp is not None and rsi < 38 and bp < 0.25


# ─── Backtest engine ──────────────────────────────────────────────────────────────
def run_backtest():
    print("Fetching data (60-90 sec)...")

    index_df = None
    try:
        idx = yf.Ticker(INDEX_TICKER).history(start=FETCH_START, end=FETCH_END)
        if not idx.empty:
            idx.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in idx.index])
            index_df = idx
            print(f"  Index {INDEX_TICKER}: {len(idx)} bars")
    except Exception as e:
        print(f"  Index fetch failed: {e}")

    all_data, valid_stocks = {}, []
    for sym in TA125_UNIVERSE:
        try:
            df = yf.Ticker(sym).history(start=FETCH_START, end=FETCH_END)
            if df.empty or len(df) < 50: continue
            df.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in df.index])
            all_data[sym] = df; valid_stocks.append(sym)
        except Exception: pass
    print(f"  Stocks: {len(valid_stocks)} loaded\n")

    all_dates = set()
    for df in all_data.values(): all_dates.update(df.index.tolist())
    trading_days = sorted(d for d in all_dates if SIM_START <= d <= SIM_END)
    if not trading_days: print("No trading days."); return

    # Benchmark
    bm_gross = bm_net = None
    if index_df is not None:
        sim_idx = index_df[(index_df.index >= SIM_START) & (index_df.index <= SIM_END)]
        if len(sim_idx) >= 2:
            p0, p1 = float(sim_idx["Close"].iloc[0]), float(sim_idx["Close"].iloc[-1])
            bm_gross = (p1/p0 - 1)*100
            gain = INITIAL_CASH*(p1/p0 - 1)
            bm_net = (gain - max(0, gain*TAX_RATE)) / INITIAL_CASH * 100

    cash = INITIAL_CASH
    positions: dict[str, Position] = {}
    trades: list[Trade] = []
    total_tax = 0.0
    regime_log = []
    last_regime = None

    # Track daily portfolio value for utilization stats
    daily_utilization = []

    for day_idx, day in enumerate(trading_days):
        day_str = day.strftime("%Y-%m-%d")
        regime = classify_regime(index_df, day) if index_df is not None else "BULL"
        (atr_mult, init_stop, tp_pct, min_bull_buy, min_bear_exit,
         max_pos, pos_pct, min_hold, rs_min, mean_rev) = REGIME_PARAMS[regime]

        if regime != last_regime:
            regime_log.append((day_str, regime))
            last_regime = regime

        # Update trail highs
        for sym, pos in positions.items():
            if sym in all_data and day in all_data[sym].index:
                cp = float(all_data[sym].loc[day, "Close"])
                if cp > pos.trail_high: pos.trail_high = cp

        # ── SELLS ───────────────────────────────────────────────────────────
        to_sell = []
        for sym, pos in list(positions.items()):
            if sym not in all_data or day not in all_data[sym].index: continue
            cp = float(all_data[sym].loc[day, "Close"])
            chg = (cp - pos.avg_cost) / pos.avg_cost
            hold_days = day_idx - pos.entry_day_idx

            # Use params from entry regime for consistency
            e_atr_mult, e_init_stop, e_tp_pct, _, e_min_bear, _, _, e_min_hold, _, _ = \
                REGIME_PARAMS[pos.regime_at_buy]

            df_sl = all_data[sym][all_data[sym].index <= day].tail(260)
            sig   = compute_signals(df_sl, 3, e_min_bear)
            atr   = compute_atr(df_sl)

            floor      = pos.avg_cost * (1 - e_init_stop)
            trail_stop = pos.trail_high - atr * e_atr_mult if atr else floor
            eff_stop   = max(trail_stop, floor)

            # 1. Trail stop (always active)
            if cp <= eff_stop:
                pct  = chg * 100
                peak = (pos.trail_high / pos.avg_cost - 1) * 100
                atr_s = f"{atr:.1f}" if atr else "n/a"
                to_sell.append((sym, f"TRAIL_STOP ({pct:+.1f}%, peak+{peak:.1f}%, ATR={atr_s})", sig))
                continue

            # 2. Bear regime — immediate exit regardless of hold time
            if regime == "BEAR":
                to_sell.append((sym, f"BEAR_EXIT (regime=BEAR)", sig))
                continue

            # 3. Take profit — only after min hold AND signal weakens
            if chg >= e_tp_pct and hold_days >= e_min_hold and sig["bias"] in ("BEARISH", "NEUTRAL"):
                to_sell.append((sym, f"TAKE_PROFIT ({chg*100:.1f}%, held {hold_days}d)", sig))
                continue

            # 4. Bearish signal — only after min hold AND requires strong bearish
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
                signals_bear=sig.get("bearish",0), regime=regime))
            del positions[sym]

        # Track utilization
        pv = cash + sum(
            p.quantity * float(all_data[p.symbol].loc[day,"Close"])
            for p in positions.values()
            if p.symbol in all_data and day in all_data[p.symbol].index
        )
        invested = pv - cash
        daily_utilization.append(invested / pv * 100 if pv > 0 else 0)

        # ── BUYS ────────────────────────────────────────────────────────────
        open_slots = max_pos - len(positions)
        if open_slots <= 0 or cash < 500 or regime == "BEAR":
            continue

        candidates = []
        for sym in valid_stocks:
            if sym in positions: continue
            if sym not in all_data or day not in all_data[sym].index: continue
            df_sl = all_data[sym][all_data[sym].index <= day].tail(260)
            sig   = compute_signals(df_sl, min_bull_buy, 3)

            if mean_rev:
                # NEUTRAL: deep dip + oversold
                if not is_mean_reversion_entry(sig): continue
                if sig["bias"] == "BEARISH": continue
                candidates.append((sym, sig["bullish"]-sig["bearish"], 0.0, sig))
            else:
                # BULL: trend + strong RS
                if sig["bias"] != "BULLISH": continue
                if sig["bullish"] < min_bull_buy: continue
                rs = relative_strength(all_data[sym], index_df, day) if index_df is not None else 0.0
                if rs is None or rs < rs_min: continue
                candidates.append((sym, sig["bullish"]-sig["bearish"], rs, sig))

        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

        for sym, score, rs, sig in candidates[:open_slots]:
            bp = float(all_data[sym].loc[day, "Close"])
            if bp <= 0: continue
            portfolio_val = cash + sum(
                p.quantity * float(all_data[p.symbol].loc[day,"Close"])
                for p in positions.values()
                if p.symbol in all_data and day in all_data[p.symbol].index
            )
            max_val = portfolio_val * pos_pct
            qty = int(min(max_val, cash * 0.98) / (bp * (1 + COMMISSION)))
            if qty < 1: continue
            outlay = qty * bp * (1 + COMMISSION)
            if outlay > cash: continue
            bc = qty * bp * COMMISSION
            cash -= outlay
            positions[sym] = Position(symbol=sym, quantity=qty, avg_cost=bp,
                buy_commission=bc, trail_high=bp,
                regime_at_buy=regime, entry_day_idx=day_idx)
            tag = f"MR" if mean_rev else f"RS+{rs:.1f}%"
            trades.append(Trade(date=day_str, symbol=sym, side="BUY",
                quantity=qty, price=round(bp,3), commission=round(bc,2),
                note=f"[{regime}|{tag}|{sig['bullish']}B] " + "; ".join(sig["bull_details"][:3]),
                signals_bull=sig["bullish"], signals_bear=sig["bearish"], regime=regime))

    # ── Final valuation ──────────────────────────────────────────────────────
    last_day = trading_days[-1]
    open_det, pos_mkt = [], 0.0
    for sym, pos in positions.items():
        df = all_data.get(sym)
        lp = float(df.loc[last_day,"Close"]) if df is not None and last_day in df.index else pos.avg_cost
        mv = pos.quantity * lp
        up = mv - (pos.quantity * pos.avg_cost + pos.buy_commission)
        pos_mkt += mv
        open_det.append((sym, pos, lp, mv, up, (lp/pos.avg_cost-1)*100))

    total_val    = cash + pos_mkt
    total_ret    = (total_val - INITIAL_CASH) / INITIAL_CASH * 100
    realized_net = sum(t.net_pnl for t in trades if t.side == "SELL")
    unrealized   = sum(r[4] for r in open_det)
    avg_util     = sum(daily_utilization) / len(daily_utilization) if daily_utilization else 0

    # ═══════════════════════════════════════════════════════════════════════════
    W = 115
    print("\n" + "="*W)
    print("  TA-125 BACKTEST v4 — INDEX-BEATING STRATEGY  |  April 2021 - March 2026")
    print(f"  BULL: 10 pos×12%, ATR 3.5x, TP 35%, min hold 15d, RS>3%, 4+ bull signals")
    print(f"  NEUTRAL: mean-reversion dips (RSI<38+lower BB), 5 pos×8%, TP 12%")
    print(f"  BEAR: immediate full exit, sit in cash")
    print("="*W)

    # Regime timeline
    print("\n  REGIME TIMELINE")
    print(f"  {'-'*W}")
    labels = {"BULL":"BULL    [trend, RS>3%, hold 15d, TP 35%]",
              "NEUTRAL":"NEUTRAL [mean-reversion dips, TP 12%]",
              "BEAR":"BEAR    [full exit, cash only]"}
    for ds, rg in regime_log:
        print(f"  {ds}  -->  {labels[rg]}")
    print(f"  Average capital utilization: {avg_util:.1f}%")

    # Trade log
    print(f"\n  {'='*W}")
    print("  TRADE LOG")
    print(f"  {'-'*W}")
    print(f"  {'Date':<13} {'Symbol':<10} {'Side':<6} {'Qty':>5} {'Price':>10}  "
          f"{'Comm':>7}  {'Gross P&L':>11}  {'Tax':>9}  {'Net P&L':>11}  {'Rgm':<8} Note")
    print(f"  {'-'*W}")
    for t in trades:
        st = "BUY  " if t.side=="BUY" else "SELL "
        ps = f"NIS{t.gross_pnl:>+10,.2f}" if t.side=="SELL" else " "*14
        ts = f"NIS{t.tax:>8,.2f}"          if t.side=="SELL" and t.tax>0 else " "*12
        ns = f"NIS{t.net_pnl:>+10,.2f}"    if t.side=="SELL" else " "*14
        nt = (t.note[:52]+"..") if len(t.note)>54 else t.note
        print(f"  {t.date:<13} {t.symbol:<10} {st}  {t.quantity:>5}  NIS{t.price:>8,.3f}  "
              f"NIS{t.commission:>5,.2f}  {ps}  {ts}  {ns}  {t.regime:<8} {nt}")

    # Open positions
    print(f"\n  {'='*W}")
    print(f"  FINAL OPEN POSITIONS  (as of {last_day.strftime('%Y-%m-%d')})")
    print(f"  {'-'*W}")
    if open_det:
        print(f"  {'Symbol':<10} {'Qty':>5}  {'AvgCost':>10}  {'Last':>10}  "
              f"{'MktVal':>11}  {'Unreal P&L':>12}  {'Ret%':>7}  {'Hold':>5}  Rgm")
        print(f"  {'-'*W}")
        for sym, pos, lp, mv, up, upct in open_det:
            hold = last_day - trading_days[pos.entry_day_idx]
            print(f"  {sym:<10} {pos.quantity:>5}  NIS{pos.avg_cost:>8,.3f}  NIS{lp:>8,.3f}  "
                  f"NIS{mv:>9,.2f}  NIS{up:>+10,.2f}  {upct:>+6.1f}%  {hold.days:>4}d  {pos.regime_at_buy}")
    else:
        print("  No open positions — fully in cash.")

    # P&L by stock
    print(f"\n  {'='*W}")
    print("  REALIZED P&L BY STOCK")
    print(f"  {'-'*W}")
    print(f"  {'Symbol':<10}  {'Gross P&L':>12}  {'Comm':>10}  {'Tax':>10}  {'Net P&L':>12}  Trades  AvgHold")
    print(f"  {'-'*W}")
    traded = sorted({t.symbol for t in trades if t.side=="SELL"})
    sg_t=sc_t=st_t=sn_t=0.0
    for sym in traded:
        sells=[t for t in trades if t.symbol==sym and t.side=="SELL"]
        buys =[t for t in trades if t.symbol==sym and t.side=="BUY"]
        sg=sum(t.gross_pnl for t in sells); sc=sum(t.commission for t in sells)+sum(t.commission for t in buys)
        st=sum(t.tax for t in sells);       sn=sum(t.net_pnl for t in sells)
        sg_t+=sg; sc_t+=sc; st_t+=st; sn_t+=sn
        print(f"  {sym:<10}  NIS{sg:>+10,.2f}  NIS{sc:>8,.2f}  NIS{st:>8,.2f}  NIS{sn:>+10,.2f}  {len(sells):>5}rt")
    if traded:
        print(f"  {'-'*W}")
        print(f"  {'SUBTOTAL':<10}  NIS{sg_t:>+10,.2f}  NIS{sc_t:>8,.2f}  NIS{st_t:>8,.2f}  NIS{sn_t:>+10,.2f}")

    # Portfolio summary
    total_comm = sum(t.commission for t in trades)
    sell_all   = [t for t in trades if t.side=="SELL"]
    wins       = [t for t in sell_all if t.net_pnl>0]
    losses     = [t for t in sell_all if t.net_pnl<=0]
    win_rate   = len(wins)/len(sell_all)*100 if sell_all else 0
    avg_win    = sum(t.net_pnl for t in wins)/len(wins)   if wins   else 0
    avg_loss   = sum(t.net_pnl for t in losses)/len(losses) if losses else 0

    print(f"\n  {'='*W}")
    print("  PORTFOLIO SUMMARY")
    print(f"  {'='*W}")
    print(f"  Starting Capital:              NIS {INITIAL_CASH:>12,.2f}")
    print(f"  Cash on Hand:                  NIS {cash:>12,.2f}")
    print(f"  Open Positions (Market Value): NIS {pos_mkt:>12,.2f}")
    print(f"  TOTAL PORTFOLIO VALUE:         NIS {total_val:>12,.2f}")
    print(f"  {'─'*60}")
    print(f"  Total Return (net of tax):         {total_ret:>+10.2f}%  (NIS {total_val-INITIAL_CASH:>+,.2f})")
    print(f"  {'─'*60}")
    print(f"  Realized Net P&L:              NIS {realized_net:>+12,.2f}")
    print(f"  Unrealized P&L (open pos):     NIS {unrealized:>+12,.2f}")
    print(f"  Combined P&L:                  NIS {realized_net+unrealized:>+12,.2f}")
    print(f"  {'─'*60}")
    print(f"  Total Commissions Paid:        NIS {total_comm:>12,.2f}")
    print(f"  Total Capital Gains Tax Paid:  NIS {total_tax:>12,.2f}")
    print(f"  Average Capital Utilization:        {avg_util:>9.1f}%")
    print(f"  {'─'*60}")
    print(f"  Buy / Sell Orders:             {len([t for t in trades if t.side=='BUY']):>6} / {len(sell_all):<6}")
    print(f"  Win Rate:                           {win_rate:>9.0f}%")
    print(f"  Avg Win / Avg Loss:            NIS {avg_win:>+7,.0f} / NIS {avg_loss:>+7,.0f}")
    if wins and losses:
        print(f"  Risk/Reward Ratio:                     {abs(avg_win/avg_loss):>6.2f}x")
    print(f"  Open Positions at End:         {len(positions):>14}")

    # Benchmark comparison
    print(f"\n  {'='*W}")
    print("  FULL COMPARISON  |  April 2021 - March 2026  |  NIS 100,000 starting capital")
    print(f"  {'-'*W}")
    print(f"  {'Strategy':<55} {'Return':>9}  {'WinRate':>8}  {'Trades':>7}  {'Avg Util':>9}")
    print(f"  {'-'*W}")
    rows = [
        ("v1  Baseline (fixed 5% stop, no filters)",         9.57, 43, 63,  "~45%"),
        ("v2  Enhanced (RS filter + ATR trailing)",           9.52, 54, 50,  "~45%"),
        ("v3  Adaptive Regime (3 modes)",                    12.08, 58, 50,  "~50%"),
        (f"v4  Index-Beating (this run)",    total_ret, int(win_rate), len(sell_all), f"{avg_util:.0f}%"),
    ]
    if bm_net is not None:
        rows.append((f"     Buy & Hold TA-125 (gross {bm_gross:+.1f}%, net after 25% tax)", bm_net, 0, 0, "100%"))
    for name,ret,wr,tr,ut in rows:
        wr_s = f"{wr:>7}%" if wr else "      -"
        tr_s = f"{tr:>7}"  if tr else "      -"
        print(f"  {name:<55} {ret:>+8.2f}%  {wr_s}  {tr_s}  {ut:>9}")
    print(f"  {'='*W}\n")


if __name__ == "__main__":
    run_backtest()
