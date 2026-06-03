"""
TA-125 Technical Analysis Backtest — Full Capital Deployment (v5)
Period: April 2021 - March 2026 (5 years)

v4 problem: avg 33% capital utilization → idle cash earns 0%

v5 solution — Cash Rotation by Regime:
  BULL    → 100% momentum stocks (same as v4)
  NEUTRAL → stocks + 40% bond ETF + 10% SPEN.TA (S&P 500 on TASE)
  BEAR    → full stock exit + 50% SPEN.TA (global hedge) + 30% bond ETF

ETF instruments used:
  SPEN.TA — קסם S&P 500 (NIS-denominated ETF on TASE)
  BOND    — simulated Israeli gov bond: ~1.5% (2021), ~1% (2022), ~4.3% (2023-26)

Why bonds earn less in 2022: rising rates caused price decline that offset coupon.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
import yfinance as yf
import pandas as pd
import ta

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────────
INITIAL_CASH   = 100_000.0
COMMISSION     = 0.0008
TAX_RATE       = 0.25
RS_LOOKBACK    = 63
ETF_COMMISSION = 0.0005     # lower commission for ETFs

INDEX_TICKER   = "^TA125.TA"
SP500_ETF      = "SPEN.TA"   # קסם S&P 500 — NIS-denominated on TASE

SIM_START    = pd.Timestamp("2021-04-01")
SIM_END      = pd.Timestamp("2026-03-31")
FETCH_START  = "2020-01-01"
FETCH_END    = "2026-04-01"

# Israeli gov bond approximate annual yield by year (net after duration risk)
BOND_ANNUAL_YIELD = {2021: 0.015, 2022: 0.010, 2023: 0.043, 2024: 0.040, 2025: 0.038, 2026: 0.038}

# Regime params: (atr_mult, init_stop, tp_pct, min_bull, min_bear, max_pos, pos_pct, min_hold, rs_min, mean_rev)
REGIME_PARAMS = {
    "BULL":    (3.5, 0.08, 0.35, 4, 4, 10, 0.12, 15, 3.0, False),
    "NEUTRAL": (2.0, 0.06, 0.12, 4, 3,  5, 0.08,  5, 0.0,  True),
    "BEAR":    (1.0, 0.05, 0.10, 9, 2,  0, 0.05,  0, 0.0, False),
}

# ETF allocation of idle cash per regime: (spen_pct, bond_pct)
# BEAR: no equity ETF (S&P 500 also falls in global bear markets), bonds only
REGIME_ETF_ALLOC = {
    "BULL":    (0.00, 0.00),   # all residual stays cash (stocks soak it up)
    "NEUTRAL": (0.10, 0.40),   # 10% SPEN (only if momentum +), 40% bonds
    "BEAR":    (0.00, 0.60),   # 0% SPEN, 60% bonds — avoid global equity in bear
}

MIN_CASH_RESERVE_PCT = 0.05   # keep 5% of portfolio as cash buffer

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
    symbol: str; quantity: int; avg_cost: float
    buy_commission: float; trail_high: float
    regime_at_buy: str; entry_day_idx: int

@dataclass
class EtfHolding:
    ticker: str       # "SPEN.TA" or "BOND"
    quantity: float   # shares for SPEN, NIS principal for BOND
    avg_cost: float   # cost per share (SPEN) or 1.0 (BOND)
    buy_commission: float

@dataclass
class Trade:
    date: str; symbol: str; side: str; quantity: float; price: float
    commission: float; gross_pnl: float = 0.0; taxable_pnl: float = 0.0
    tax: float = 0.0; net_pnl: float = 0.0
    note: str = ""; signals_bull: int = 0; signals_bear: int = 0
    regime: str = ""; category: str = "STOCK"  # STOCK | ETF | BOND


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
    if c > sma200 and slope > 0.008 and (adx or 0) > 22: return "BULL"
    if c < sma200 and slope < -0.008:                     return "BEAR"
    return "NEUTRAL"

def bond_daily_return(day: pd.Timestamp) -> float:
    yr = BOND_ANNUAL_YIELD.get(day.year, 0.040)
    return yr / 252   # ~252 trading days

def relative_strength(stock_df, index_df, day, lookback=RS_LOOKBACK):
    sh = stock_df[stock_df.index <= day]["Close"]
    ih = index_df[index_df.index <= day]["Close"]
    if len(sh) < lookback + 1 or len(ih) < lookback + 1: return None
    return round((float(sh.iloc[-1])/float(sh.iloc[-lookback]) - 1)*100 -
                 (float(ih.iloc[-1])/float(ih.iloc[-lookback]) - 1)*100, 2)

def compute_signals(df_slice, min_bull, min_bear):
    if len(df_slice) < 30:
        return {"bullish":0,"bearish":0,"bias":"INSUFFICIENT","rsi":None,
                "bull_details":[],"bear_details":[],"bb_pct":None}
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


# ─── ETF management ───────────────────────────────────────────────────────────────
def etf_price(ticker: str, day: pd.Timestamp, etf_data: dict) -> float | None:
    if ticker == "BOND": return 1.0  # bond is modeled as principal, not price
    df = etf_data.get(ticker)
    if df is None: return None
    hist = df[df.index <= day]
    if hist.empty: return None
    return float(hist["Close"].iloc[-1])

def sell_etf(holding: EtfHolding, sell_price: float, day_str: str,
             regime: str, trades: list, cash_ref: list, tax_ref: list):
    """Sell an ETF holding and update cash. Returns net cash received."""
    if holding.ticker == "BOND":
        # Bond: principal + accrued interest
        gross = holding.quantity - holding.avg_cost * holding.quantity  # shouldn't happen
        # Actually bond quantity IS the NIS value (principal + interest accrued)
        principal_cost = holding.avg_cost  # original principal
        gross_pnl = holding.quantity - principal_cost
        sell_comm  = holding.quantity * ETF_COMMISSION
        taxable    = gross_pnl - holding.buy_commission - sell_comm
        tax        = max(0.0, taxable * TAX_RATE)
        net_recv   = holding.quantity - sell_comm - tax
    else:
        gross_pnl  = (sell_price - holding.avg_cost) * holding.quantity
        sell_comm  = holding.quantity * sell_price * ETF_COMMISSION
        taxable    = gross_pnl - holding.buy_commission - sell_comm
        tax        = max(0.0, taxable * TAX_RATE)
        net_recv   = holding.quantity * sell_price - sell_comm - tax

    cash_ref[0] += net_recv
    tax_ref[0]  += tax
    trades.append(Trade(
        date=day_str, symbol=holding.ticker, side="SELL",
        quantity=round(holding.quantity, 2),
        price=round(sell_price, 4) if holding.ticker != "BOND" else round(holding.quantity, 2),
        commission=round(sell_comm, 2),
        gross_pnl=round(gross_pnl, 2), taxable_pnl=round(taxable, 2),
        tax=round(tax, 2), net_pnl=round(net_recv - (holding.quantity * sell_price - sell_comm if holding.ticker != "BOND" else holding.quantity - sell_comm), 2),
        note=f"ETF_SELL (regime={regime})", regime=regime,
        category="BOND" if holding.ticker == "BOND" else "ETF",
    ))


# ─── Backtest engine ──────────────────────────────────────────────────────────────
def run_backtest():
    print("Fetching data (60-90 sec)...")

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

    # S&P 500 ETF (SPEN.TA)
    etf_data: dict[str, pd.DataFrame] = {}
    try:
        sp = yf.Ticker(SP500_ETF).history(start=FETCH_START, end=FETCH_END)
        if not sp.empty:
            sp.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in sp.index])
            etf_data[SP500_ETF] = sp
            print(f"  {SP500_ETF} (S&P 500 ETF): {len(sp)} bars")
    except Exception as e:
        print(f"  {SP500_ETF} fetch failed: {e}")

    # Stocks
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
    if index_df is not None: all_dates.update(index_df.index.tolist())
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

    # ── Portfolio state ────────────────────────────────────────────────────────
    cash            = INITIAL_CASH
    positions:      dict[str, Position]    = {}
    etf_holdings:   dict[str, EtfHolding]  = {}  # "SPEN.TA" | "BOND"
    trades:         list[Trade]            = []
    total_tax       = 0.0
    regime_log      = []
    last_regime     = None

    # For mutable pass-by-ref in helper
    cash_ref  = [cash]
    tax_ref   = [total_tax]

    daily_utilization = []
    regime_days = {"BULL": 0, "NEUTRAL": 0, "BEAR": 0}

    # ── Simulation loop ────────────────────────────────────────────────────────
    for day_idx, day in enumerate(trading_days):
        day_str = day.strftime("%Y-%m-%d")
        regime  = classify_regime(index_df, day) if index_df is not None else "BULL"
        (atr_mult, init_stop, tp_pct, min_bull_buy, min_bear_exit,
         max_pos, pos_pct, min_hold, rs_min, mean_rev) = REGIME_PARAMS[regime]
        spen_alloc, bond_alloc = REGIME_ETF_ALLOC[regime]

        regime_days[regime] += 1
        if regime != last_regime:
            regime_log.append((day_str, regime))
            last_regime = regime

        # Sync mutable cash
        cash = cash_ref[0]

        # Update trail highs
        for sym, pos in positions.items():
            if sym in all_data and day in all_data[sym].index:
                cp = float(all_data[sym].loc[day, "Close"])
                if cp > pos.trail_high: pos.trail_high = cp

        # Accrue bond interest
        if "BOND" in etf_holdings:
            dr = bond_daily_return(day)
            etf_holdings["BOND"].quantity *= (1 + dr)

        # ── STOCK SELLS ────────────────────────────────────────────────────
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

        # ── ETF SELLS — liquidate on regime change if allocation changed ───
        new_spen, new_bond = REGIME_ETF_ALLOC[regime]
        # Sell SPEN if not needed in this regime (BULL has 0 ETF alloc)
        if new_spen == 0.0 and "SPEN.TA" in etf_holdings:
            sp_price = etf_price("SPEN.TA", day, etf_data)
            if sp_price:
                h = etf_holdings["SPEN.TA"]
                gp = (sp_price - h.avg_cost) * h.quantity
                sc = h.quantity * sp_price * ETF_COMMISSION
                tx = gp - h.buy_commission - sc
                tax = max(0.0, tx * TAX_RATE)
                cash += h.quantity * sp_price - sc - tax
                total_tax += tax
                trades.append(Trade(date=day_str, symbol="SPEN.TA", side="SELL",
                    quantity=round(h.quantity,4), price=round(sp_price,3),
                    commission=round(sc,2), gross_pnl=round(gp,2),
                    taxable_pnl=round(tx,2), tax=round(tax,2),
                    net_pnl=round(tx-tax,2),
                    note=f"ETF_LIQUIDATE (regime={regime})", regime=regime, category="ETF"))
                del etf_holdings["SPEN.TA"]

        if new_bond == 0.0 and "BOND" in etf_holdings:
            h = etf_holdings["BOND"]
            principal_cost = h.avg_cost
            gp = h.quantity - principal_cost
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

        # ── STOCK BUYS ────────────────────────────────────────────────────
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

        # ── ETF BUYS — deploy remaining idle cash ─────────────────────────
        # Calculate portfolio value
        pos_val = sum(
            p.quantity * float(all_data[p.symbol].loc[day,"Close"])
            for p in positions.values()
            if p.symbol in all_data and day in all_data[p.symbol].index
        )
        etf_val = sum(
            (etf_holdings[k].quantity * (etf_price(k, day, etf_data) or etf_holdings[k].avg_cost)
             if k != "BOND" else etf_holdings[k].quantity)
            for k in etf_holdings
        )
        portfolio_val = cash + pos_val + etf_val
        reserve       = portfolio_val * MIN_CASH_RESERVE_PCT
        deployable    = max(0.0, cash - reserve)

        # Target ETF values
        target_spen = portfolio_val * spen_alloc
        target_bond = portfolio_val * bond_alloc

        # Buy SPEN.TA if under target — only when SPEN is above its SMA50 (positive momentum)
        def spen_has_momentum(day, etf_data):
            df = etf_data.get(SP500_ETF)
            if df is None: return False
            hist = df[df.index <= day]["Close"]
            if len(hist) < 50: return False
            sma50 = float(hist.tail(50).mean())
            return float(hist.iloc[-1]) > sma50

        if spen_alloc > 0 and deployable > 500 and SP500_ETF in etf_data and spen_has_momentum(day, etf_data):
            current_spen_val = (etf_holdings["SPEN.TA"].quantity * etf_price("SPEN.TA", day, etf_data)
                                if "SPEN.TA" in etf_holdings and etf_price("SPEN.TA", day, etf_data)
                                else 0.0)
            sp_price = etf_price("SPEN.TA", day, etf_data)
            if sp_price and current_spen_val < target_spen * 0.90:
                to_invest = min(target_spen - current_spen_val, deployable * 0.8)
                qty = to_invest / (sp_price * (1 + ETF_COMMISSION))
                if qty >= 0.01:
                    bc = qty * sp_price * ETF_COMMISSION
                    cost = qty * sp_price + bc
                    if cost <= cash:
                        cash -= cost
                        if "SPEN.TA" in etf_holdings:
                            h = etf_holdings["SPEN.TA"]
                            total_qty = h.quantity + qty
                            h.avg_cost = (h.avg_cost * h.quantity + sp_price * qty) / total_qty
                            h.quantity = total_qty
                            h.buy_commission += bc
                        else:
                            etf_holdings["SPEN.TA"] = EtfHolding("SPEN.TA", qty, sp_price, bc)
                        deployable -= cost
                        trades.append(Trade(date=day_str, symbol="SPEN.TA", side="BUY",
                            quantity=round(qty,4), price=round(sp_price,3),
                            commission=round(bc,2),
                            note=f"[{regime}] S&P500 ETF rotation", regime=regime, category="ETF"))

        # Buy BOND if under target
        if bond_alloc > 0 and deployable > 500:
            current_bond_val = etf_holdings["BOND"].quantity if "BOND" in etf_holdings else 0.0
            if current_bond_val < target_bond * 0.90:
                to_invest = min(target_bond - current_bond_val, deployable * 0.8)
                bc = to_invest * ETF_COMMISSION
                cost = to_invest + bc
                if cost <= cash:
                    cash -= cost
                    if "BOND" in etf_holdings:
                        h = etf_holdings["BOND"]
                        h.avg_cost += to_invest   # track total principal invested
                        h.quantity += to_invest
                        h.buy_commission += bc
                    else:
                        etf_holdings["BOND"] = EtfHolding("BOND", to_invest, to_invest, bc)
                    trades.append(Trade(date=day_str, symbol="BOND", side="BUY",
                        quantity=round(to_invest,2), price=1.0,
                        commission=round(bc,2),
                        note=f"[{regime}] Bond ETF rotation (~{BOND_ANNUAL_YIELD.get(day.year,0.04)*100:.1f}% annual)",
                        regime=regime, category="BOND"))

        cash_ref[0] = cash

        # Utilization tracking
        pv2 = cash + pos_val + etf_val
        daily_utilization.append((pv2 - cash) / pv2 * 100 if pv2 > 0 else 0)

    # ── Final valuation ───────────────────────────────────────────────────────
    cash = cash_ref[0]
    last_day = trading_days[-1]
    open_stocks, stock_mkt = [], 0.0
    for sym, pos in positions.items():
        df = all_data.get(sym)
        lp = float(df.loc[last_day,"Close"]) if df is not None and last_day in df.index else pos.avg_cost
        mv = pos.quantity * lp
        up = mv - (pos.quantity * pos.avg_cost + pos.buy_commission)
        stock_mkt += mv
        open_stocks.append((sym, pos, lp, mv, up, (lp/pos.avg_cost-1)*100))

    open_etfs, etf_mkt = [], 0.0
    for ticker, h in etf_holdings.items():
        if ticker == "BOND":
            mv = h.quantity
            up = h.quantity - h.avg_cost
            open_etfs.append((ticker, h, 1.0, mv, up, (h.quantity/h.avg_cost-1)*100 if h.avg_cost > 0 else 0))
        else:
            lp = etf_price(ticker, last_day, etf_data) or h.avg_cost
            mv = h.quantity * lp
            up = mv - (h.quantity * h.avg_cost + h.buy_commission)
            open_etfs.append((ticker, h, lp, mv, up, (lp/h.avg_cost-1)*100))
        etf_mkt += mv

    total_val    = cash + stock_mkt + etf_mkt
    total_ret    = (total_val - INITIAL_CASH) / INITIAL_CASH * 100
    realized_net = sum(t.net_pnl for t in trades if t.side == "SELL")
    unrealized   = sum(r[4] for r in open_stocks) + sum(r[4] for r in open_etfs)
    avg_util     = sum(daily_utilization) / len(daily_utilization) if daily_utilization else 0

    # ═══════════════════════════════════════════════════════════════════════════
    W = 120
    print("\n" + "="*W)
    print("  TA-125 BACKTEST v5 — FULL CAPITAL DEPLOYMENT  |  April 2021 - March 2026  (5 years)")
    print(f"  BULL: momentum stocks | NEUTRAL: stocks + BOND 40% + SPEN 10% (if momentum+) | BEAR: BOND 60% only")
    print(f"  SPEN.TA = קסם S&P 500 (NIS, TASE) | BOND = Israel gov bond (simulated yield)")
    print("="*W)

    # Regime summary
    total_d = sum(regime_days.values())
    print(f"\n  REGIME BREAKDOWN  ({total_d} trading days)")
    print(f"  {'-'*W}")
    for rg, d in regime_days.items():
        bar = "#" * int(d/total_d*60)
        print(f"  {rg:<8} {d:>4}d ({d/total_d*100:>4.1f}%)  {bar}")
    print(f"\n  REGIME TIMELINE")
    for ds, rg in regime_log:
        labels = {"BULL":"trend stocks, RS>3%", "NEUTRAL":"mean-rev + ETF bonds", "BEAR":"global ETF rotation"}
        print(f"    {ds}  {rg}  [{labels[rg]}]")

    # P&L by stock
    print(f"\n  {'='*W}")
    print("  REALIZED P&L BY INSTRUMENT")
    print(f"  {'-'*W}")
    print(f"  {'Symbol':<12}  {'Category':<7}  {'Gross P&L':>12}  {'Comm':>8}  {'Tax':>10}  {'Net P&L':>12}  Trades")
    print(f"  {'-'*W}")
    all_symbols = sorted({t.symbol for t in trades if t.side=="SELL"})
    sg_t=sc_t=st_t=sn_t=0.0
    for sym in all_symbols:
        sells = [t for t in trades if t.symbol==sym and t.side=="SELL"]
        buys  = [t for t in trades if t.symbol==sym and t.side=="BUY"]
        cat   = sells[0].category if sells else "?"
        sg=sum(t.gross_pnl for t in sells); sc=sum(t.commission for t in sells)+sum(t.commission for t in buys)
        st=sum(t.tax for t in sells);       sn=sum(t.net_pnl for t in sells)
        sg_t+=sg; sc_t+=sc; st_t+=st; sn_t+=sn
        print(f"  {sym:<12}  {cat:<7}  NIS{sg:>+10,.2f}  NIS{sc:>6,.2f}  NIS{st:>8,.2f}  NIS{sn:>+10,.2f}  {len(sells)}rt")
    print(f"  {'-'*W}")
    print(f"  {'SUBTOTAL':<12}  {'':7}  NIS{sg_t:>+10,.2f}  NIS{sc_t:>6,.2f}  NIS{st_t:>8,.2f}  NIS{sn_t:>+10,.2f}")

    # Open positions
    print(f"\n  {'='*W}")
    print(f"  FINAL OPEN POSITIONS  (as of {last_day.strftime('%Y-%m-%d')})")
    print(f"  {'-'*W}")
    if open_stocks or open_etfs:
        print(f"  {'Symbol':<12}  {'Cat':<5}  {'Qty':>8}  {'AvgCost':>10}  {'Last':>10}  {'MktVal':>11}  {'Unreal':>11}  {'Ret%':>7}")
        print(f"  {'-'*W}")
        for sym, pos, lp, mv, up, upct in open_stocks:
            print(f"  {sym:<12}  STOCK  {pos.quantity:>8}  NIS{pos.avg_cost:>8,.2f}  NIS{lp:>8,.2f}  NIS{mv:>9,.2f}  NIS{up:>+9,.2f}  {upct:>+6.1f}%")
        for ticker, h, lp, mv, up, upct in open_etfs:
            qty_s = f"{h.quantity:>8,.0f}" if ticker == "BOND" else f"{h.quantity:>8,.2f}"
            print(f"  {ticker:<12}  {'BOND' if ticker=='BOND' else 'ETF':<5}  {qty_s}  NIS{h.avg_cost if ticker!='BOND' else h.avg_cost:>8,.2f}  {'  (yield)':>10}  NIS{mv:>9,.2f}  NIS{up:>+9,.2f}  {upct:>+6.1f}%")
    else:
        print("  No open positions.")

    # Portfolio summary
    total_comm = sum(t.commission for t in trades)
    sell_all   = [t for t in trades if t.side=="SELL" and t.category=="STOCK"]
    wins   = [t for t in sell_all if t.net_pnl > 0]
    losses = [t for t in sell_all if t.net_pnl <= 0]
    win_rate = len(wins)/len(sell_all)*100 if sell_all else 0
    avg_win  = sum(t.net_pnl for t in wins)/len(wins)   if wins   else 0
    avg_loss = sum(t.net_pnl for t in losses)/len(losses) if losses else 0

    etf_trades_sell = [t for t in trades if t.side=="SELL" and t.category in ("ETF","BOND")]
    etf_net = sum(t.net_pnl for t in etf_trades_sell)

    print(f"\n  {'='*W}")
    print("  PORTFOLIO SUMMARY")
    print(f"  {'='*W}")
    print(f"  Starting Capital:                    NIS {INITIAL_CASH:>12,.2f}")
    print(f"  Cash on Hand:                        NIS {cash:>12,.2f}")
    print(f"  Open Stock Positions (Mkt Value):    NIS {stock_mkt:>12,.2f}")
    print(f"  Open ETF/Bond Holdings (Mkt Value):  NIS {etf_mkt:>12,.2f}")
    print(f"  TOTAL PORTFOLIO VALUE:               NIS {total_val:>12,.2f}")
    print(f"  {'─'*65}")
    print(f"  Total Return (net of tax, 5 years):      {total_ret:>+10.2f}%  (NIS {total_val-INITIAL_CASH:>+,.2f})")
    print(f"  CAGR (approx):                           {((1+total_ret/100)**(1/5)-1)*100:>+10.2f}%")
    print(f"  {'─'*65}")
    print(f"  Realized Stock P&L (net):            NIS {sum(t.net_pnl for t in sell_all):>+12,.2f}")
    print(f"  Realized ETF/Bond P&L (net):         NIS {etf_net:>+12,.2f}")
    print(f"  Unrealized P&L (open pos):           NIS {unrealized:>+12,.2f}")
    print(f"  {'─'*65}")
    print(f"  Total Commissions Paid:              NIS {total_comm:>12,.2f}")
    print(f"  Total Capital Gains Tax Paid:        NIS {total_tax:>12,.2f}")
    print(f"  Average Capital Utilization:              {avg_util:>9.1f}%")
    print(f"  {'─'*65}")
    print(f"  Stock Trades (buy/sell):             {len([t for t in trades if t.side=='BUY' and t.category=='STOCK']):>6} / {len(sell_all):<6}")
    print(f"  ETF/Bond Trades (buy/sell):          {len([t for t in trades if t.side=='BUY' and t.category in ('ETF','BOND')]):>6} / {len(etf_trades_sell):<6}")
    print(f"  Stock Win Rate:                           {win_rate:>9.0f}%")
    print(f"  Avg Stock Win / Loss:                NIS {avg_win:>+7,.0f} / NIS {avg_loss:>+7,.0f}")
    if wins and losses:
        print(f"  Stock R/R Ratio:                              {abs(avg_win/avg_loss):>6.2f}x")

    # Final comparison
    print(f"\n  {'='*W}")
    print("  5-YEAR COMPARISON  |  April 2021 - March 2026  |  NIS 100,000")
    print(f"  {'-'*W}")
    print(f"  {'Strategy':<60}  {'Return':>9}  {'CAGR':>7}  {'Util':>6}  {'WinRate':>8}")
    print(f"  {'-'*W}")
    rows = [
        ("v4  Adaptive Regime (stocks only, idle cash = 0%)",        66.16, 10.7, 33,  60),
        (f"v5  Full Deployment (stocks + ETF/Bond rotation)",  total_ret, ((1+total_ret/100)**(1/5)-1)*100, avg_util, win_rate),
    ]
    if bm_net is not None:
        rows.append((f"     Buy & Hold TA-125 (gross {bm_gross:+.1f}%, net after tax)", bm_net,
                     ((1+bm_net/100)**(1/5)-1)*100, 100, 0))
    for name, ret, cagr, ut, wr in rows:
        wr_s = f"{wr:.0f}%" if wr else "  -"
        print(f"  {name:<60}  {ret:>+8.2f}%  {cagr:>+6.1f}%  {ut:>5.0f}%  {wr_s:>7}")
    print(f"  {'='*W}\n")


if __name__ == "__main__":
    run_backtest()
