"""
TA-125 Technical Analysis Backtest — Adaptive Regime Strategy (v3)
Period: May 2025 - March 2026 (TASE trading days)

Three regimes, three strategies:
  BULL    — Trend following:   wide ATR stop, hold winners, RS filter
  NEUTRAL — Mean reversion:    RSI+BB dip-buy, quick TP (7%), tight stop
  BEAR    — Defensive:         no new buys, aggressive exit, max 20% exposure

Regime detection: Index SMA200 position + SMA50 slope + ADX
Benchmark: TA-125 buy-and-hold (with 25% tax) for same period
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
import yfinance as yf
import pandas as pd
import ta

warnings.filterwarnings("ignore")

# ─── Base config ─────────────────────────────────────────────────────────────────
INITIAL_CASH = 100_000.0
COMMISSION   = 0.0008
TAX_RATE     = 0.25
RS_LOOKBACK  = 63          # relative-strength lookback (trading days)

INDEX_TICKER = "^TA125.TA"
SIM_START    = pd.Timestamp("2025-04-27")
SIM_END      = pd.Timestamp("2026-03-31")
FETCH_START  = "2024-10-01"
FETCH_END    = "2026-04-01"

# ─── Regime-adaptive parameters ──────────────────────────────────────────────────
REGIME_PARAMS = {
    #           atr_mult  initial_stop  tp_pct  min_bull  min_bear  max_pos  max_pos_pct  mean_rev
    "BULL":    (2.5,       0.07,         0.15,   3,        3,        10,      0.10,        False),
    "NEUTRAL": (1.5,       0.05,         0.07,   4,        3,        5,       0.08,        True),
    "BEAR":    (1.0,       0.05,         0.08,   5,        2,        2,       0.05,        False),
}

# ─── TA-125 universe ─────────────────────────────────────────────────────────────
TA125_UNIVERSE = [
    "POLI.TA","LUMI.TA","DSCT.TA","MZRH.TA","FIBI.TA",
    "NICE.TA","CAMT.TA","TSEM.TA","NVMI.TA","SPNS.TA","ITRN.TA",
    "ESLT.TA","TEVA.TA","ICL.TA","BEZQ.TA",
    "SKBN.TA","RSEL.TA",
    "PHNX.TA","HARL.TA","MGDL.TA","MNRN.TA",
    "AZRG.TA","AMOT.TA","ALHE.TA","ELCO.TA",
    "ENLT.TA","DLEKG.TA","BAZAN.TA","ILCO.TA",
]

# ─── Data classes ────────────────────────────────────────────────────────────────
@dataclass
class Position:
    symbol:         str
    quantity:       int
    avg_cost:       float
    buy_commission: float
    trail_high:     float
    regime_at_buy:  str = "BULL"

@dataclass
class Trade:
    date:         str
    symbol:       str
    side:         str
    quantity:     int
    price:        float
    commission:   float
    gross_pnl:    float = 0.0
    taxable_pnl:  float = 0.0
    tax:          float = 0.0
    net_pnl:      float = 0.0
    note:         str   = ""
    signals_bull: int   = 0
    signals_bear: int   = 0
    regime:       str   = ""


# ─── Helpers ─────────────────────────────────────────────────────────────────────
def _last(s: pd.Series):
    v = s.iloc[-1]; return float(v) if pd.notna(v) else None

def _prev(s: pd.Series):
    v = s.iloc[-2] if len(s) > 1 else None
    return float(v) if v is not None and pd.notna(v) else None


def classify_regime(index_df: pd.DataFrame, day: pd.Timestamp) -> str:
    """Classify market regime as BULL / NEUTRAL / BEAR using index data up to `day`."""
    hist = index_df[index_df.index <= day].tail(260)
    if len(hist) < 60:
        return "BULL"   # not enough history — assume bull

    close = hist["Close"]
    high  = hist["High"]  if "High"  in hist.columns else close
    low   = hist["Low"]   if "Low"   in hist.columns else close

    sma200_s = ta.trend.sma_indicator(close, window=200)
    sma50_s  = ta.trend.sma_indicator(close, window=50)
    adx_s    = ta.trend.adx(high, low, close, window=14)

    sma200 = _last(sma200_s)
    sma50  = _last(sma50_s)
    adx    = _last(adx_s)
    c      = float(close.iloc[-1])

    # SMA50 slope: % change over last 30 bars
    if len(sma50_s.dropna()) >= 31:
        sma50_now  = float(sma50_s.dropna().iloc[-1])
        sma50_30d  = float(sma50_s.dropna().iloc[-31])
        slope = (sma50_now - sma50_30d) / sma50_30d if sma50_30d else 0.0
    else:
        slope = 0.0

    if sma200 is None:
        return "BULL"

    above_200 = c > sma200
    rising    = slope > 0.008        # >0.8% monthly slope
    falling   = slope < -0.008
    trending  = (adx or 0) > 22

    if above_200 and rising and trending:
        return "BULL"
    elif (not above_200) and (falling or trending):
        return "BEAR"
    else:
        return "NEUTRAL"


def compute_atr(df_slice: pd.DataFrame, window: int = 14) -> float | None:
    if len(df_slice) < window + 1:
        return None
    s = ta.volatility.average_true_range(df_slice["High"], df_slice["Low"], df_slice["Close"], window=window)
    v = s.iloc[-1]; return float(v) if pd.notna(v) else None


def compute_signals(df_slice: pd.DataFrame, min_bull: int, min_bear: int) -> dict:
    if len(df_slice) < 30:
        return {"bullish": 0, "bearish": 0, "bias": "INSUFFICIENT",
                "rsi": None, "bull_details": [], "bear_details": [],
                "bb_pct": None, "close": None}

    close = df_slice["Close"]; high = df_slice["High"]; low = df_slice["Low"]

    sma20  = ta.trend.sma_indicator(close, 20)
    sma50  = ta.trend.sma_indicator(close, 50)
    sma200 = ta.trend.sma_indicator(close, 200) if len(df_slice) >= 200 \
             else pd.Series([float("nan")] * len(df_slice), index=close.index)
    ema12  = ta.trend.ema_indicator(close, 12)
    ema26  = ta.trend.ema_indicator(close, 26)
    rsi14  = ta.momentum.rsi(close, 14)

    macd_o  = ta.trend.MACD(close, 12, 26, 9)
    macd_l  = macd_o.macd(); macd_s = macd_o.macd_signal()

    st_o  = ta.momentum.StochasticOscillator(high, low, close, 14, smooth_window=3)
    sk    = st_o.stoch(); sd = st_o.stoch_signal()

    bb_o  = ta.volatility.BollingerBands(close, 20, window_dev=2)
    bbu   = bb_o.bollinger_hband(); bbl = bb_o.bollinger_lband()

    c       = float(close.iloc[-1])
    v_s20   = _last(sma20);  v_s50 = _last(sma50); v_s200 = _last(sma200)
    v_e12   = _last(ema12);  v_e26 = _last(ema26)
    v_rsi   = _last(rsi14)
    v_ml    = _last(macd_l); v_ms  = _last(macd_s)
    pv_ml   = _prev(macd_l); pv_ms = _prev(macd_s)
    v_sk    = _last(sk);     v_sd  = _last(sd)
    v_bbu   = _last(bbu);    v_bbl = _last(bbl)

    bb_pct = None
    if v_bbu and v_bbl:
        bw = v_bbu - v_bbl
        bb_pct = (c - v_bbl) / bw if bw > 0 else 0.5

    bull, bear = [], []

    if v_rsi is not None:
        if   v_rsi < 30:  bull.append(f"RSI oversold ({v_rsi:.1f})")
        elif v_rsi < 40:  bull.append(f"RSI low zone ({v_rsi:.1f})")
        elif v_rsi > 70:  bear.append(f"RSI overbought ({v_rsi:.1f})")
        elif v_rsi > 60:  bear.append(f"RSI high zone ({v_rsi:.1f})")

    if all(x is not None for x in [v_ml, v_ms, pv_ml, pv_ms]):
        if   v_ml > v_ms and pv_ml <= pv_ms: bull.append("MACD bullish crossover")
        elif v_ml < v_ms and pv_ml >= pv_ms: bear.append("MACD bearish crossover")
        (bull if v_ml > v_ms else bear).append(f"MACD {'above' if v_ml>v_ms else 'below'} signal")

    if v_s20:  (bull if c > v_s20  else bear).append(f"Price {'above' if c>v_s20  else 'below'} SMA20")
    if v_s50:  (bull if c > v_s50  else bear).append(f"Price {'above' if c>v_s50  else 'below'} SMA50")
    if v_s200 and not pd.isna(v_s200):
        (bull if c > v_s200 else bear).append(f"Price {'above' if c>v_s200 else 'below'} SMA200")
    if v_e12 and v_e26:
        (bull if v_e12 > v_e26 else bear).append(f"EMA12 {'>' if v_e12>v_e26 else '<'} EMA26")

    if v_bbu and v_bbl and bb_pct is not None:
        if   c < v_bbl:      bull.append("Below lower BB")
        elif c > v_bbu:      bear.append("Above upper BB")
        elif bb_pct < 0.30:  bull.append(f"Lower BB zone ({bb_pct*100:.0f}%)")
        elif bb_pct > 0.70:  bear.append(f"Upper BB zone ({bb_pct*100:.0f}%)")

    if v_sk is not None and v_sd is not None:
        if   v_sk < 25 and v_sd < 25: bull.append(f"Stoch oversold (K={v_sk:.0f})")
        elif v_sk > 75 and v_sd > 75: bear.append(f"Stoch overbought (K={v_sk:.0f})")

    nb, nb2 = len(bull), len(bear)
    if nb >= min_bull and nb > nb2 + 1:   bias = "BULLISH"
    elif nb2 >= min_bear and nb2 > nb + 1: bias = "BEARISH"
    else:                                   bias = "NEUTRAL"

    return {"bullish": nb, "bearish": nb2, "bias": bias,
            "rsi": round(v_rsi, 1) if v_rsi else None,
            "close": c, "bb_pct": bb_pct,
            "bull_details": bull, "bear_details": bear}


def is_mean_reversion_entry(sig: dict) -> bool:
    """NEUTRAL regime entry: RSI oversold AND price in lower BB zone."""
    rsi     = sig.get("rsi") or 999
    bb_pct  = sig.get("bb_pct")
    if bb_pct is None:
        return False
    return rsi < 40 and bb_pct < 0.30


def relative_strength(stock_df, index_df, day, lookback=RS_LOOKBACK):
    sh = stock_df[stock_df.index <= day]["Close"]
    ih = index_df[index_df.index <= day]["Close"]
    if len(sh) < lookback + 1 or len(ih) < lookback + 1:
        return None
    sr = (float(sh.iloc[-1]) / float(sh.iloc[-lookback]) - 1) * 100
    ir = (float(ih.iloc[-1]) / float(ih.iloc[-lookback]) - 1) * 100
    return round(sr - ir, 2)


# ─── Backtest engine ─────────────────────────────────────────────────────────────
def run_backtest():
    print("Fetching TASE data + index (this may take 60-90 seconds)...")

    # ── Index ──────────────────────────────────────────────────────────────────
    index_df = None
    try:
        t = yf.Ticker(INDEX_TICKER)
        idx = t.history(start=FETCH_START, end=FETCH_END)
        if not idx.empty:
            idx.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in idx.index])
            index_df = idx
            print(f"  Index {INDEX_TICKER}: {len(idx)} bars")
    except Exception as e:
        print(f"  WARNING: index fetch failed: {e}")

    # ── Stocks ─────────────────────────────────────────────────────────────────
    all_data: dict[str, pd.DataFrame] = {}
    valid_stocks: list[str] = []
    for sym in TA125_UNIVERSE:
        try:
            df = yf.Ticker(sym).history(start=FETCH_START, end=FETCH_END)
            if df.empty or len(df) < 50:
                continue
            df.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in df.index])
            all_data[sym] = df
            valid_stocks.append(sym)
        except Exception:
            pass
    print(f"  Stocks: {len(valid_stocks)} loaded\n")

    # ── Trading days ───────────────────────────────────────────────────────────
    all_dates: set = set()
    for df in all_data.values():
        all_dates.update(df.index.tolist())
    trading_days = sorted(d for d in all_dates if SIM_START <= d <= SIM_END)
    if not trading_days:
        print("No trading days found."); return

    # ── Benchmark: index buy-and-hold ──────────────────────────────────────────
    benchmark_return = None
    benchmark_net    = None
    if index_df is not None:
        idx_sim = index_df[(index_df.index >= SIM_START) & (index_df.index <= SIM_END)]
        if len(idx_sim) >= 2:
            p0  = float(idx_sim["Close"].iloc[0])
            p1  = float(idx_sim["Close"].iloc[-1])
            gross_gain   = INITIAL_CASH * (p1 / p0 - 1)
            tax_bm       = max(0, gross_gain * TAX_RATE)
            net_gain     = gross_gain - tax_bm
            benchmark_return = (p1 / p0 - 1) * 100
            benchmark_net    = net_gain / INITIAL_CASH * 100

    # ── Portfolio state ────────────────────────────────────────────────────────
    cash:           float               = INITIAL_CASH
    positions:      dict[str, Position] = {}
    trades:         list[Trade]         = []
    total_tax_paid: float               = 0.0

    regime_log:     list[tuple[str, str]] = []  # (date, regime)
    last_logged_regime = None

    # ── Main loop ─────────────────────────────────────────────────────────────
    for day in trading_days:
        day_str = day.strftime("%Y-%m-%d")

        # Regime detection
        regime = classify_regime(index_df, day) if index_df is not None else "BULL"
        (atr_mult, init_stop, tp_pct, min_bull, min_bear,
         max_pos, max_pos_pct, mean_rev_mode) = REGIME_PARAMS[regime]

        if regime != last_logged_regime:
            regime_log.append((day_str, regime))
            last_logged_regime = regime

        # Update trailing highs
        for sym, pos in positions.items():
            if sym in all_data and day in all_data[sym].index:
                cp = float(all_data[sym].loc[day, "Close"])
                if cp > pos.trail_high:
                    pos.trail_high = cp

        # ── SELLS ──────────────────────────────────────────────────────────
        to_sell: list[tuple[str, str, dict]] = []

        for sym, pos in list(positions.items()):
            if sym not in all_data or day not in all_data[sym].index:
                continue

            cp         = float(all_data[sym].loc[day, "Close"])
            change_pct = (cp - pos.avg_cost) / pos.avg_cost

            # Use the params from when the position was opened (avoid mid-trade regime changes)
            p_atr_mult, p_init_stop, p_tp_pct, p_min_bull, p_min_bear, _, _, _ = \
                REGIME_PARAMS[pos.regime_at_buy]

            df_slice = all_data[sym][all_data[sym].index <= day].tail(260)
            sig      = compute_signals(df_slice, p_min_bull, p_min_bear)
            atr_val  = compute_atr(df_slice)

            init_floor  = pos.avg_cost * (1 - p_init_stop)
            trail_stop  = pos.trail_high - atr_val * p_atr_mult if atr_val else init_floor
            eff_stop    = max(trail_stop, init_floor)

            if cp <= eff_stop:
                pct    = (cp - pos.avg_cost) / pos.avg_cost * 100
                peak_p = (pos.trail_high - pos.avg_cost) / pos.avg_cost * 100
                atr_s  = f"{atr_val:.1f}" if atr_val else "n/a"
                to_sell.append((sym, f"TRAIL_STOP ({pct:+.1f}%, peak+{peak_p:.1f}%, ATR={atr_s})", sig))
                continue

            # In BEAR regime: exit on NEUTRAL or BEARISH signal (more aggressive)
            if regime == "BEAR" and sig["bias"] in ("BEARISH", "NEUTRAL"):
                to_sell.append((sym, f"BEAR_REGIME_EXIT ({sig['bearish']}b/{sig['bullish']}B)", sig))
                continue

            if change_pct >= p_tp_pct and sig["bias"] in ("BEARISH", "NEUTRAL"):
                to_sell.append((sym, f"TAKE_PROFIT ({change_pct*100:.1f}%) + signal weakened", sig))
                continue

            if sig["bias"] == "BEARISH":
                to_sell.append((sym, f"BEARISH ({sig['bearish']}b/{sig['bullish']}B)", sig))

        for sym, reason, sig in to_sell:
            pos       = positions[sym]
            sp        = float(all_data[sym].loc[day, "Close"])
            sell_comm = pos.quantity * sp * COMMISSION
            gross     = (sp - pos.avg_cost) * pos.quantity
            taxable   = gross - pos.buy_commission - sell_comm
            tax       = max(0.0, taxable * TAX_RATE)
            net       = taxable - tax

            cash           += pos.quantity * sp - sell_comm - tax
            total_tax_paid += tax
            trades.append(Trade(
                date=day_str, symbol=sym, side="SELL",
                quantity=pos.quantity, price=round(sp, 3),
                commission=round(sell_comm, 2),
                gross_pnl=round(gross, 2), taxable_pnl=round(taxable, 2),
                tax=round(tax, 2), net_pnl=round(net, 2),
                note=reason, signals_bull=sig.get("bullish", 0),
                signals_bear=sig.get("bearish", 0), regime=regime,
            ))
            del positions[sym]

        # ── BUYS ───────────────────────────────────────────────────────────
        if regime == "BEAR":
            continue   # no new positions in bear market

        open_slots = max_pos - len(positions)
        if open_slots <= 0 or cash < 200:
            continue

        candidates: list[tuple[str, int, float, dict]] = []

        for sym in valid_stocks:
            if sym in positions:
                continue
            if sym not in all_data or day not in all_data[sym].index:
                continue

            df_slice = all_data[sym][all_data[sym].index <= day].tail(260)
            sig      = compute_signals(df_slice, min_bull, min_bear)

            if mean_rev_mode:
                # NEUTRAL: mean reversion entry — RSI<40 + lower BB zone
                if not is_mean_reversion_entry(sig):
                    continue
                # Still require at least NEUTRAL bias (not bearish)
                if sig["bias"] == "BEARISH":
                    continue
                score = sig["bullish"] - sig["bearish"]
                candidates.append((sym, score, 0.0, sig))
            else:
                # BULL: trend-following + relative strength
                if sig["bias"] != "BULLISH":
                    continue
                rs = relative_strength(all_data[sym], index_df, day) if index_df is not None else 0.0
                if rs is None or rs < 0:
                    continue
                score = sig["bullish"] - sig["bearish"]
                candidates.append((sym, score, rs, sig))

        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

        for sym, score, rs, sig in candidates[:open_slots]:
            bp = float(all_data[sym].loc[day, "Close"])
            if bp <= 0:
                continue

            portfolio_value = cash + sum(
                p.quantity * float(all_data[p.symbol].loc[day, "Close"])
                for p in positions.values()
                if p.symbol in all_data and day in all_data[p.symbol].index
            )
            max_value = portfolio_value * max_pos_pct
            qty = int(min(max_value, cash * 0.98) / (bp * (1 + COMMISSION)))
            if qty < 1:
                continue
            outlay = qty * bp * (1 + COMMISSION)
            if outlay > cash:
                continue

            buy_comm = qty * bp * COMMISSION
            cash    -= outlay
            positions[sym] = Position(
                symbol=sym, quantity=qty, avg_cost=bp,
                buy_commission=buy_comm, trail_high=bp,
                regime_at_buy=regime,
            )

            mode_tag = "MR" if mean_rev_mode else f"RS+{rs:.1f}%"
            note_str = f"[{regime}|{mode_tag}] " + "; ".join(sig["bull_details"][:3])
            trades.append(Trade(
                date=day_str, symbol=sym, side="BUY",
                quantity=qty, price=round(bp, 3),
                commission=round(buy_comm, 2),
                note=note_str, signals_bull=sig["bullish"],
                signals_bear=sig["bearish"], regime=regime,
            ))

    # ── Final valuation ───────────────────────────────────────────────────────
    last_day = trading_days[-1]
    open_detail = []
    pos_mkt_val = 0.0
    for sym, pos in positions.items():
        df = all_data.get(sym)
        lp = float(df.loc[last_day, "Close"]) if df is not None and last_day in df.index else pos.avg_cost
        mv = pos.quantity * lp
        upnl = mv - (pos.quantity * pos.avg_cost + pos.buy_commission)
        pos_mkt_val += mv
        open_detail.append((sym, pos, lp, mv, upnl, (lp - pos.avg_cost) / pos.avg_cost * 100))

    total_value  = cash + pos_mkt_val
    total_return = (total_value - INITIAL_CASH) / INITIAL_CASH * 100
    realized_net = sum(t.net_pnl for t in trades if t.side == "SELL")
    unrealized   = sum(r[4] for r in open_detail)

    # ═══════════════════════════════════════════════════════════════════════════
    # OUTPUT
    # ═══════════════════════════════════════════════════════════════════════════
    W = 115

    print("\n" + "=" * W)
    print("  TA-125 BACKTEST v3 — ADAPTIVE REGIME STRATEGY  |  TASE  |  May 2025 - March 2026")
    print(f"  Universe: {len(valid_stocks)} stocks  |  Capital: NIS 100,000")
    print(f"  BULL: trend+RS filter, ATR 2.5x, TP 15%  |  "
          f"NEUTRAL: mean-reversion, ATR 1.5x, TP 7%  |  "
          f"BEAR: defensive, no new buys")
    print("=" * W)

    # Regime timeline
    print("\n  REGIME TIMELINE")
    print(f"  {'-'*W}")
    for date_str, reg in regime_log:
        label = {"BULL": "BULL    [trend following, RS filter, wide stop]",
                 "NEUTRAL": "NEUTRAL [mean reversion, tight stop, quick TP]",
                 "BEAR":    "BEAR    [defensive — no new buys, aggressive exit]"}[reg]
        print(f"  {date_str}  -->  {label}")

    # Trade log
    print(f"\n  {'='*W}")
    print("  TRADE LOG")
    print(f"  {'-'*W}")
    print(f"  {'Date':<13} {'Symbol':<10} {'Side':<6} {'Qty':>5} {'Price':>10}  "
          f"{'Comm':>7}  {'Gross P&L':>11}  {'Tax':>9}  {'Net P&L':>11}  {'Rgm':<7}  Note")
    print(f"  {'-'*W}")

    for t in trades:
        st  = "BUY  " if t.side == "BUY" else "SELL "
        ps  = f"NIS{t.gross_pnl:>+10,.2f}" if t.side == "SELL" else " " * 14
        ts  = f"NIS{t.tax:>8,.2f}"          if t.side == "SELL" and t.tax > 0 else " " * 12
        ns  = f"NIS{t.net_pnl:>+10,.2f}"    if t.side == "SELL" else " " * 14
        nt  = (t.note[:52] + "..") if len(t.note) > 54 else t.note
        print(f"  {t.date:<13} {t.symbol:<10} {st}  {t.quantity:>5}  NIS{t.price:>8,.3f}  "
              f"NIS{t.commission:>5,.2f}  {ps}  {ts}  {ns}  {t.regime:<7}  {nt}")

    if not trades:
        print("  No trades executed.")

    # Open positions
    print(f"\n  {'='*W}")
    print(f"  FINAL OPEN POSITIONS  (as of {last_day.strftime('%Y-%m-%d')})")
    print(f"  {'-'*W}")
    if open_detail:
        print(f"  {'Symbol':<10} {'Qty':>5}  {'Avg Cost':>10}  {'Last':>10}  "
              f"{'Mkt Val':>11}  {'Unreal P&L':>12}  {'Ret%':>7}  Rgm")
        print(f"  {'-'*W}")
        for sym, pos, lp, mv, upnl, upct in open_detail:
            print(f"  {sym:<10} {pos.quantity:>5}  NIS{pos.avg_cost:>8,.3f}  NIS{lp:>8,.3f}  "
                  f"NIS{mv:>9,.2f}  NIS{upnl:>+10,.2f}  {upct:>+6.1f}%  {pos.regime_at_buy}")
    else:
        print("  No open positions — fully in cash.")

    # P&L by stock
    print(f"\n  {'='*W}")
    print("  REALIZED P&L BY STOCK")
    print(f"  {'-'*W}")
    print(f"  {'Symbol':<10}  {'Gross P&L':>12}  {'Commissions':>13}  {'Tax':>10}  {'Net P&L':>12}  Trades")
    print(f"  {'-'*W}")
    traded = sorted({t.symbol for t in trades if t.side == "SELL"})
    sg_tot = sc_tot = st_tot = sn_tot = 0.0
    for sym in traded:
        sells = [t for t in trades if t.symbol == sym and t.side == "SELL"]
        buys  = [t for t in trades if t.symbol == sym and t.side == "BUY"]
        sg = sum(t.gross_pnl   for t in sells)
        sc = sum(t.commission  for t in sells) + sum(t.commission for t in buys)
        st = sum(t.tax         for t in sells)
        sn = sum(t.net_pnl     for t in sells)
        sg_tot += sg; sc_tot += sc; st_tot += st; sn_tot += sn
        print(f"  {sym:<10}  NIS{sg:>+10,.2f}  NIS{sc:>11,.2f}  NIS{st:>8,.2f}  NIS{sn:>+10,.2f}  {len(sells)} rt")
    if traded:
        print(f"  {'-'*W}")
        print(f"  {'SUBTOTAL':<10}  NIS{sg_tot:>+10,.2f}  NIS{sc_tot:>11,.2f}  NIS{st_tot:>8,.2f}  NIS{sn_tot:>+10,.2f}")

    # Portfolio summary
    total_comm = sum(t.commission for t in trades)
    sell_all   = [t for t in trades if t.side == "SELL"]
    wins       = [t for t in sell_all if t.net_pnl > 0]
    losses     = [t for t in sell_all if t.net_pnl <= 0]
    win_rate   = len(wins) / len(sell_all) * 100 if sell_all else 0
    avg_win    = sum(t.net_pnl for t in wins)   / len(wins)   if wins   else 0
    avg_loss   = sum(t.net_pnl for t in losses) / len(losses) if losses else 0

    print(f"\n  {'='*W}")
    print("  PORTFOLIO SUMMARY")
    print(f"  {'='*W}")
    print(f"  Starting Capital:              NIS {INITIAL_CASH:>12,.2f}")
    print(f"  Cash on Hand:                  NIS {cash:>12,.2f}")
    print(f"  Open Positions (Market Value): NIS {pos_mkt_val:>12,.2f}")
    print(f"  TOTAL PORTFOLIO VALUE:         NIS {total_value:>12,.2f}")
    print(f"  {'─'*60}")
    print(f"  Total Return (net of tax):         {total_return:>+10.2f}%  (NIS {total_value-INITIAL_CASH:>+,.2f})")
    print(f"  {'─'*60}")
    print(f"  Realized Net P&L:              NIS {realized_net:>+12,.2f}")
    print(f"  Unrealized P&L (open pos):     NIS {unrealized:>+12,.2f}")
    print(f"  {'─'*60}")
    print(f"  Total Commissions Paid:        NIS {total_comm:>12,.2f}")
    print(f"  Total Capital Gains Tax Paid:  NIS {total_tax_paid:>12,.2f}")
    print(f"  {'─'*60}")
    print(f"  Total Buy Orders:              {len([t for t in trades if t.side=='BUY']):>14}")
    print(f"  Total Sell Orders:             {len(sell_all):>14}")
    print(f"  Win Rate (closed trades):      {win_rate:>13.0f}%")
    print(f"  Avg Win:                       NIS {avg_win:>+12,.2f}")
    print(f"  Avg Loss:                      NIS {avg_loss:>+12,.2f}")
    if wins and losses:
        rr = abs(avg_win / avg_loss)
        print(f"  Risk/Reward Ratio:             {rr:>14.2f}x")
    print(f"  Open Positions at End:         {len(positions):>14}")

    # ── BENCHMARK COMPARISON ──────────────────────────────────────────────────
    print(f"\n  {'='*W}")
    print("  BENCHMARK COMPARISON  (same period: May 2025 - March 2026)")
    print(f"  {'-'*W}")

    strategies = [
        ("v1  Baseline (fixed 5% stop)",            9.57,  43, 63,  1.99),
        ("v2  Enhanced (RS + ATR trailing)",         9.52,  54, 50,  1.40),
        (f"v3  Adaptive Regime (this run)",   total_return, int(win_rate), len(sell_all),
         abs(avg_win / avg_loss) if wins and losses else 0),
    ]
    if benchmark_return is not None:
        strategies.append((
            f"BUY & HOLD TA-125 index (gross {benchmark_return:+.1f}%, net after 25% tax)",
            benchmark_net, "-", "-", "-"
        ))

    print(f"  {'Strategy':<50} {'Return':>9}  {'WinRate':>8}  {'Trades':>7}  {'R/R':>6}")
    print(f"  {'-'*W}")
    for name, ret, wr, tr, rr in strategies:
        wr_s = f"{wr:>7}%" if isinstance(wr, int) else f"{wr:>7}"
        tr_s = f"{tr:>7}"  if isinstance(tr, int) else f"{tr:>7}"
        rr_s = f"{rr:>5.2f}x" if isinstance(rr, float) and rr > 0 else f"{rr:>6}"
        print(f"  {name:<50} {ret:>+8.2f}%  {wr_s}  {tr_s}  {rr_s}")

    print(f"  {'='*W}\n")


if __name__ == "__main__":
    run_backtest()
