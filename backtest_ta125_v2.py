"""
TA-125 Technical Analysis Backtest — Enhanced Strategy (v2)
Period: May 2025 - March 2026 (TASE trading days)
Strategy: Pure TA + three alpha enhancements:
  1. Market Regime Filter  — buy only when TA-125 index is above its SMA200
  2. Relative Strength     — buy only stocks outperforming the index over 63 days
  3. ATR Trailing Stop     — dynamic stop that rises with price (replaces fixed 5%)
Rules: Long-only, max 10% per position, no short selling
Costs: 0.08% commission, 25% capital gains tax on realized profits
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass
import yfinance as yf
import pandas as pd
import numpy as np
import ta

warnings.filterwarnings("ignore")

# ─── Configuration ──────────────────────────────────────────────────────────────
INITIAL_CASH       = 100_000.0
COMMISSION         = 0.0008        # 0.08% each way
TAX_RATE           = 0.25          # 25% on realized gains
MAX_POS_PCT        = 0.10          # Max 10% per position
INITIAL_STOP_PCT   = 0.07          # Hard floor: max 7% loss from entry (initial stop)
ATR_TRAIL_MULT     = 2.5           # Trailing stop = highest_price - ATR*mult
TAKE_PROFIT_PCT    = 0.15          # Take-profit only if signal also weakens
MIN_BULLISH        = 3
MIN_BEARISH        = 3
MAX_POSITIONS      = 10
RS_LOOKBACK        = 63            # Relative-strength lookback in trading days (~3 months)

INDEX_TICKER = "^TA125.TA"

SIM_START   = pd.Timestamp("2025-04-27")
SIM_END     = pd.Timestamp("2026-03-31")
FETCH_START = "2024-10-01"
FETCH_END   = "2026-04-01"

# ─── TA-125 Universe ─────────────────────────────────────────────────────────────
TA125_UNIVERSE = [
    "POLI.TA", "LUMI.TA", "DSCT.TA", "MZRH.TA", "FIBI.TA",
    "NICE.TA", "CAMT.TA", "TSEM.TA", "NVMI.TA", "SPNS.TA", "ITRN.TA",
    "ESLT.TA",
    "TEVA.TA",
    "ICL.TA",
    "BEZQ.TA",
    "SKBN.TA", "RSEL.TA",
    "PHNX.TA", "HARL.TA", "MGDL.TA", "MNRN.TA",
    "AZRG.TA", "AMOT.TA", "ALHE.TA", "ELCO.TA",
    "ENLT.TA", "DLEKG.TA",
    "BAZAN.TA", "ILCO.TA",
]


# ─── Data classes ────────────────────────────────────────────────────────────────
@dataclass
class Position:
    symbol:        str
    quantity:      int
    avg_cost:      float
    buy_commission: float
    trail_high:    float   # highest close since entry (for ATR trailing stop)


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
    exit_reason:  str   = ""


# ─── Helpers ─────────────────────────────────────────────────────────────────────
def _last(s: pd.Series):
    v = s.iloc[-1]
    return float(v) if pd.notna(v) else None

def _prev(s: pd.Series):
    v = s.iloc[-2] if len(s) > 1 else None
    return float(v) if v is not None and pd.notna(v) else None


def compute_atr(df_slice: pd.DataFrame, window: int = 14) -> float | None:
    """Return latest ATR(14) value, or None if insufficient data."""
    if len(df_slice) < window + 1:
        return None
    atr_series = ta.volatility.average_true_range(
        df_slice["High"], df_slice["Low"], df_slice["Close"], window=window
    )
    v = atr_series.iloc[-1]
    return float(v) if pd.notna(v) else None


def compute_signals(df_slice: pd.DataFrame) -> dict:
    """Compute all TA signals on a historical slice ending at today."""
    if len(df_slice) < 30:
        return {"bullish": 0, "bearish": 0, "bias": "INSUFFICIENT",
                "rsi": None, "bull_details": [], "bear_details": []}

    close = df_slice["Close"]
    high  = df_slice["High"]
    low   = df_slice["Low"]

    sma20  = ta.trend.sma_indicator(close, window=20)
    sma50  = ta.trend.sma_indicator(close, window=50)
    sma200 = ta.trend.sma_indicator(close, window=200) if len(df_slice) >= 200 \
             else pd.Series([float("nan")] * len(df_slice), index=close.index)
    ema12  = ta.trend.ema_indicator(close, window=12)
    ema26  = ta.trend.ema_indicator(close, window=26)
    rsi14  = ta.momentum.rsi(close, window=14)

    macd_obj    = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    macd_line   = macd_obj.macd()
    macd_signal = macd_obj.macd_signal()

    stoch_obj = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    stoch_k   = stoch_obj.stoch()
    stoch_d   = stoch_obj.stoch_signal()

    bb_obj   = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = bb_obj.bollinger_hband()
    bb_lower = bb_obj.bollinger_lband()

    c      = float(close.iloc[-1])
    v_sma20  = _last(sma20);  v_sma50  = _last(sma50);  v_sma200 = _last(sma200)
    v_ema12  = _last(ema12);  v_ema26  = _last(ema26)
    v_rsi    = _last(rsi14)
    v_macd   = _last(macd_line);    v_macd_s  = _last(macd_signal)
    pv_macd  = _prev(macd_line);    pv_macd_s = _prev(macd_signal)
    v_sk     = _last(stoch_k);      v_sd      = _last(stoch_d)
    v_bbu    = _last(bb_upper);     v_bbl     = _last(bb_lower)

    bull, bear = [], []

    if v_rsi is not None:
        if v_rsi < 30:   bull.append(f"RSI oversold ({v_rsi:.1f})")
        elif v_rsi < 40: bull.append(f"RSI low zone ({v_rsi:.1f})")
        elif v_rsi > 70: bear.append(f"RSI overbought ({v_rsi:.1f})")
        elif v_rsi > 60: bear.append(f"RSI high zone ({v_rsi:.1f})")

    if all(x is not None for x in [v_macd, v_macd_s, pv_macd, pv_macd_s]):
        if v_macd > v_macd_s and pv_macd <= pv_macd_s:
            bull.append("MACD bullish crossover")
        elif v_macd < v_macd_s and pv_macd >= pv_macd_s:
            bear.append("MACD bearish crossover")
        (bull if v_macd > v_macd_s else bear).append(
            f"MACD {'above' if v_macd > v_macd_s else 'below'} signal"
        )

    if v_sma20:  (bull if c > v_sma20  else bear).append(f"Price {'above' if c>v_sma20  else 'below'} SMA20")
    if v_sma50:  (bull if c > v_sma50  else bear).append(f"Price {'above' if c>v_sma50  else 'below'} SMA50")
    if v_sma200 and not pd.isna(v_sma200):
        (bull if c > v_sma200 else bear).append(f"Price {'above' if c>v_sma200 else 'below'} SMA200")

    if v_ema12 and v_ema26:
        (bull if v_ema12 > v_ema26 else bear).append(
            f"EMA12 {'>' if v_ema12 > v_ema26 else '<'} EMA26"
        )

    if v_bbu and v_bbl:
        bw = v_bbu - v_bbl
        if bw > 0:
            bp = (c - v_bbl) / bw
            if   c < v_bbl:    bull.append("Below lower BB")
            elif c > v_bbu:    bear.append("Above upper BB")
            elif bp < 0.30:    bull.append(f"Lower BB zone ({bp*100:.0f}%)")
            elif bp > 0.70:    bear.append(f"Upper BB zone ({bp*100:.0f}%)")

    if v_sk is not None and v_sd is not None:
        if   v_sk < 25 and v_sd < 25: bull.append(f"Stoch oversold (K={v_sk:.0f})")
        elif v_sk > 75 and v_sd > 75: bear.append(f"Stoch overbought (K={v_sk:.0f})")

    nb, nb2 = len(bull), len(bear)
    if nb >= MIN_BULLISH and nb > nb2 + 1:
        bias = "BULLISH"
    elif nb2 >= MIN_BEARISH and nb2 > nb + 1:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return {
        "bullish": nb, "bearish": nb2, "bias": bias,
        "rsi": round(v_rsi, 1) if v_rsi else None,
        "close": c, "bull_details": bull, "bear_details": bear,
    }


def compute_index_sma200(index_df: pd.DataFrame, day: pd.Timestamp) -> tuple[float | None, float | None]:
    """Return (index_close, sma200) for the given day."""
    hist = index_df[index_df.index <= day]
    if len(hist) < 5:
        return None, None
    c = float(hist["Close"].iloc[-1])
    sma = ta.trend.sma_indicator(hist["Close"], window=200)
    s = _last(sma)
    return c, s


def relative_strength(stock_df: pd.DataFrame, index_df: pd.DataFrame,
                       day: pd.Timestamp, lookback: int = RS_LOOKBACK) -> float | None:
    """
    Return stock's excess return vs index over last `lookback` trading days.
    Positive = stock outperforming the index.
    """
    stock_hist = stock_df[stock_df.index <= day]["Close"]
    index_hist = index_df[index_df.index <= day]["Close"]

    if len(stock_hist) < lookback + 1 or len(index_hist) < lookback + 1:
        return None

    stock_ret = (float(stock_hist.iloc[-1]) / float(stock_hist.iloc[-lookback]) - 1) * 100
    index_ret = (float(index_hist.iloc[-1]) / float(index_hist.iloc[-lookback]) - 1) * 100
    return round(stock_ret - index_ret, 2)


# ─── Backtest engine ─────────────────────────────────────────────────────────────
def run_backtest():
    print("Fetching TASE data + index (this may take 60-90 seconds)...")

    # ── Fetch index ────────────────────────────────────────────────────────────
    index_df: pd.DataFrame | None = None
    try:
        t = yf.Ticker(INDEX_TICKER)
        idx = t.history(start=FETCH_START, end=FETCH_END)
        if not idx.empty:
            idx.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in idx.index])
            index_df = idx
            print(f"  Index {INDEX_TICKER}: {len(idx)} bars loaded.")
    except Exception as e:
        print(f"  WARNING: could not fetch index {INDEX_TICKER}: {e}")
        print("  Market regime filter will be DISABLED.")

    # ── Fetch stocks ───────────────────────────────────────────────────────────
    all_data: dict[str, pd.DataFrame] = {}
    valid_stocks: list[str] = []

    for sym in TA125_UNIVERSE:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(start=FETCH_START, end=FETCH_END)
            if df.empty or len(df) < 50:
                continue
            df.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in df.index])
            all_data[sym] = df
            valid_stocks.append(sym)
        except Exception:
            pass

    if not valid_stocks:
        print("ERROR: Could not fetch any stock data.")
        return

    print(f"  Stocks loaded: {len(valid_stocks)} — {', '.join(valid_stocks)}\n")

    # ── Trading days ───────────────────────────────────────────────────────────
    all_dates: set = set()
    for df in all_data.values():
        all_dates.update(df.index.tolist())
    trading_days = sorted(d for d in all_dates if SIM_START <= d <= SIM_END)

    if not trading_days:
        print("No trading days found in range.")
        return

    # ── Portfolio state ────────────────────────────────────────────────────────
    cash:            float               = INITIAL_CASH
    positions:       dict[str, Position] = {}
    trades:          list[Trade]         = []
    total_tax_paid:  float               = 0.0
    regime_blocks:   int                 = 0   # days where regime filter blocked buys

    # ── Simulation loop ────────────────────────────────────────────────────────
    for day in trading_days:
        day_str = day.strftime("%Y-%m-%d")

        # ── Update trailing highs ──────────────────────────────────────────
        for sym, pos in positions.items():
            if sym in all_data and day in all_data[sym].index:
                cp = float(all_data[sym].loc[day, "Close"])
                if cp > pos.trail_high:
                    pos.trail_high = cp

        # ── SELLS: Check exit conditions ───────────────────────────────────
        to_sell: list[tuple[str, str, dict]] = []

        for sym, pos in list(positions.items()):
            if sym not in all_data or day not in all_data[sym].index:
                continue

            current_price = float(all_data[sym].loc[day, "Close"])
            change_pct    = (current_price - pos.avg_cost) / pos.avg_cost

            df_slice = all_data[sym][all_data[sym].index <= day].tail(260)
            sig      = compute_signals(df_slice)

            # 1. ATR Trailing Stop
            atr = compute_atr(df_slice)
            initial_floor = pos.avg_cost * (1 - INITIAL_STOP_PCT)

            if atr is not None:
                trail_stop  = pos.trail_high - atr * ATR_TRAIL_MULT
                eff_stop    = max(trail_stop, initial_floor)
            else:
                eff_stop    = initial_floor

            if current_price <= eff_stop:
                stop_pct = (current_price - pos.avg_cost) / pos.avg_cost * 100
                trail_pct = (pos.trail_high - pos.avg_cost) / pos.avg_cost * 100
                atr_str = f"{atr:.1f}" if atr else "n/a"
                note = (f"TRAIL_STOP ({stop_pct:+.1f}%,  "
                        f"peak+{trail_pct:.1f}%,  "
                        f"ATR={atr_str})")
                to_sell.append((sym, note, sig))
                continue

            # 2. Take profit + signal weakens
            if change_pct >= TAKE_PROFIT_PCT and sig["bias"] in ("BEARISH", "NEUTRAL"):
                to_sell.append((sym, f"TAKE_PROFIT ({change_pct*100:.1f}%) + signal weakened", sig))
                continue

            # 3. Bearish signal
            if sig["bias"] == "BEARISH":
                to_sell.append((sym, f"BEARISH ({sig['bearish']}bear/{sig['bullish']}bull)", sig))

        # Execute sells
        for sym, reason, sig in to_sell:
            pos        = positions[sym]
            sell_price = float(all_data[sym].loc[day, "Close"])
            sell_comm  = pos.quantity * sell_price * COMMISSION
            gross_pnl  = (sell_price - pos.avg_cost) * pos.quantity
            taxable    = gross_pnl - pos.buy_commission - sell_comm
            tax        = max(0.0, taxable * TAX_RATE)
            net_pnl    = taxable - tax

            cash           += pos.quantity * sell_price - sell_comm - tax
            total_tax_paid += tax

            trades.append(Trade(
                date=day_str, symbol=sym, side="SELL",
                quantity=pos.quantity, price=round(sell_price, 3),
                commission=round(sell_comm, 2),
                gross_pnl=round(gross_pnl, 2), taxable_pnl=round(taxable, 2),
                tax=round(tax, 2), net_pnl=round(net_pnl, 2),
                note=reason,
                signals_bull=sig.get("bullish", 0),
                signals_bear=sig.get("bearish", 0),
            ))
            del positions[sym]

        # ── BUYS ───────────────────────────────────────────────────────────
        open_slots = MAX_POSITIONS - len(positions)
        if open_slots <= 0 or cash < 200:
            continue

        # Enhancement 1 — Market Regime Filter
        if index_df is not None:
            idx_close, idx_sma200 = compute_index_sma200(index_df, day)
            if idx_close is not None and idx_sma200 is not None:
                if idx_close < idx_sma200:
                    regime_blocks += 1
                    continue   # Index in downtrend → no new buys today

        # Scan for buy candidates
        buy_candidates: list[tuple[str, int, float, dict]] = []

        for sym in valid_stocks:
            if sym in positions:
                continue
            if sym not in all_data or day not in all_data[sym].index:
                continue

            df_slice = all_data[sym][all_data[sym].index <= day].tail(260)
            sig      = compute_signals(df_slice)
            if sig["bias"] != "BULLISH":
                continue

            # Enhancement 2 — Relative Strength Filter
            if index_df is not None:
                rs = relative_strength(all_data[sym], index_df, day)
                if rs is None or rs < 0:
                    continue   # stock lagging the index → skip
            else:
                rs = 0.0

            score = sig["bullish"] - sig["bearish"]
            buy_candidates.append((sym, score, rs, sig))

        # Sort: primary = TA score, secondary = relative strength
        buy_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

        for sym, score, rs, sig in buy_candidates[:open_slots]:
            buy_price = float(all_data[sym].loc[day, "Close"])
            if buy_price <= 0:
                continue

            portfolio_value = cash + sum(
                p.quantity * float(all_data[p.symbol].loc[day, "Close"])
                for p in positions.values()
                if p.symbol in all_data and day in all_data[p.symbol].index
            )
            max_value = portfolio_value * MAX_POS_PCT
            quantity  = int(min(max_value, cash * 0.98) / (buy_price * (1 + COMMISSION)))
            if quantity < 1:
                continue

            buy_comm    = quantity * buy_price * COMMISSION
            total_outlay = quantity * buy_price + buy_comm
            if total_outlay > cash:
                continue

            cash -= total_outlay
            positions[sym] = Position(
                symbol=sym, quantity=quantity,
                avg_cost=buy_price, buy_commission=buy_comm,
                trail_high=buy_price,
            )

            rs_note   = f"RS+{rs:.1f}%" if rs else ""
            top_sigs  = sig["bull_details"][:3]
            note_str  = "; ".join(filter(None, [rs_note] + top_sigs))

            trades.append(Trade(
                date=day_str, symbol=sym, side="BUY",
                quantity=quantity, price=round(buy_price, 3),
                commission=round(buy_comm, 2),
                note=note_str,
                signals_bull=sig["bullish"], signals_bear=sig["bearish"],
            ))

    # ── Final valuation ───────────────────────────────────────────────────────
    last_day = trading_days[-1]
    open_positions_detail = []
    positions_mkt_value   = 0.0

    for sym, pos in positions.items():
        df        = all_data.get(sym)
        last_price = float(df.loc[last_day, "Close"]) if df is not None and last_day in df.index \
                     else pos.avg_cost
        mkt_val    = pos.quantity * last_price
        unreal_pnl = mkt_val - (pos.quantity * pos.avg_cost + pos.buy_commission)
        unreal_pct = (last_price - pos.avg_cost) / pos.avg_cost * 100
        positions_mkt_value += mkt_val
        open_positions_detail.append((sym, pos, last_price, mkt_val, unreal_pnl, unreal_pct))

    total_value  = cash + positions_mkt_value
    total_return = (total_value - INITIAL_CASH) / INITIAL_CASH * 100
    realized_net = sum(t.net_pnl for t in trades if t.side == "SELL")
    unrealized   = sum(row[4] for row in open_positions_detail)

    # ─── OUTPUT ──────────────────────────────────────────────────────────────────
    W = 110
    print("\n" + "=" * W)
    print("  TA-125 BACKTEST v2 — ENHANCED STRATEGY  |  TASE  |  May 2025 - March 2026")
    print(f"  Universe: {len(valid_stocks)} stocks  |  Capital: NIS 100,000")
    print(f"  Enhancements: Market Regime (SMA200)  |  Relative Strength (63d)  |  ATR Trailing Stop ({ATR_TRAIL_MULT}x)")
    print(f"  Commissions: 0.08%  |  Capital Gains Tax: 25%  |  Max Position: 10%")
    print(f"  Initial Stop: {INITIAL_STOP_PCT*100:.0f}%  |  Regime-blocked days: {regime_blocks}")
    print("=" * W)

    # Trade Log
    print("\n  TRADE LOG")
    print(f"  {'-'*W}")
    print(f"  {'Date':<13} {'Symbol':<10} {'Side':<6} {'Qty':>6} {'Price':>10}  {'Comm':>8}  "
          f"{'Gross P&L':>11}  {'Tax':>9}  {'Net P&L':>11}  Signals  Note")
    print(f"  {'-'*W}")

    for t in trades:
        side_tag = "BUY  " if t.side == "BUY" else "SELL "
        pnl_str  = f"NIS{t.gross_pnl:>+10,.2f}" if t.side == "SELL" else " " * 14
        tax_str  = f"NIS{t.tax:>8,.2f}"          if t.side == "SELL" and t.tax > 0 else " " * 12
        net_str  = f"NIS{t.net_pnl:>+10,.2f}"    if t.side == "SELL" else " " * 14
        sig_str  = f"{t.signals_bull}B/{t.signals_bear}b"
        note_trim = (t.note[:50] + "..") if len(t.note) > 52 else t.note
        print(f"  {t.date:<13} {t.symbol:<10} {side_tag}  {t.quantity:>6}  NIS{t.price:>8,.3f}  "
              f"NIS{t.commission:>6,.2f}  {pnl_str}  {tax_str}  {net_str}  {sig_str:<8}  {note_trim}")

    if not trades:
        print("  No trades executed in the simulation period.")

    # Open Positions
    print(f"\n  {'='*W}")
    print(f"  FINAL OPEN POSITIONS  (as of {last_day.strftime('%Y-%m-%d')})")
    print(f"  {'-'*W}")

    if open_positions_detail:
        print(f"  {'Symbol':<10} {'Qty':>6}  {'Avg Cost':>10}  {'Last Price':>11}  {'Trail High':>11}  "
              f"{'Mkt Value':>11}  {'Unreal P&L':>12}  {'Ret %':>8}  Status")
        print(f"  {'-'*W}")
        for sym, pos, lp, mv, upnl, upct in open_positions_detail:
            status = "PROFIT" if upnl > 0 else "LOSS  "
            print(f"  {sym:<10} {pos.quantity:>6}  NIS{pos.avg_cost:>8,.3f}  NIS{lp:>9,.3f}  "
                  f"NIS{pos.trail_high:>9,.3f}  NIS{mv:>9,.2f}  NIS{upnl:>+10,.2f}  "
                  f"{upct:>+7.2f}%  {status}")
    else:
        print("  No open positions — fully in cash.")

    # Realized P&L by stock
    print(f"\n  {'='*W}")
    print("  REALIZED P&L BY STOCK (Closed Trades Only)")
    print(f"  {'-'*W}")
    print(f"  {'Symbol':<10}  {'Gross P&L':>12}  {'Commissions':>13}  {'Tax':>10}  {'Net P&L':>12}  {'Trades':>8}")
    print(f"  {'-'*W}")

    traded_syms = sorted({t.symbol for t in trades if t.side == "SELL"})
    sum_gross = sum_comm = sum_tax = sum_net = 0.0

    for sym in traded_syms:
        sells = [t for t in trades if t.symbol == sym and t.side == "SELL"]
        buys  = [t for t in trades if t.symbol == sym and t.side == "BUY"]
        sg  = sum(t.gross_pnl for t in sells)
        sc  = sum(t.commission for t in sells) + sum(t.commission for t in buys)
        st  = sum(t.tax for t in sells)
        sn  = sum(t.net_pnl for t in sells)
        sum_gross += sg; sum_comm += sc; sum_tax += st; sum_net += sn
        print(f"  {sym:<10}  NIS{sg:>+10,.2f}  NIS{sc:>11,.2f}  NIS{st:>8,.2f}  NIS{sn:>+10,.2f}  {len(sells):>8} rt")

    if traded_syms:
        print(f"  {'-'*W}")
        print(f"  {'SUBTOTAL':<10}  NIS{sum_gross:>+10,.2f}  NIS{sum_comm:>11,.2f}  NIS{sum_tax:>8,.2f}  NIS{sum_net:>+10,.2f}")

    # Portfolio Summary
    total_comm  = sum(t.commission for t in trades)
    sell_all    = [t for t in trades if t.side == "SELL"]
    wins        = [t for t in sell_all if t.net_pnl > 0]
    losses      = [t for t in sell_all if t.net_pnl <= 0]
    win_rate    = len(wins) / len(sell_all) * 100 if sell_all else 0
    avg_win     = sum(t.net_pnl for t in wins)   / len(wins)   if wins   else 0
    avg_loss    = sum(t.net_pnl for t in losses) / len(losses) if losses else 0

    print(f"\n  {'='*W}")
    print("  PORTFOLIO SUMMARY — ENHANCED STRATEGY v2")
    print(f"  {'='*W}")
    print(f"  Starting Capital:              NIS {INITIAL_CASH:>12,.2f}")
    print(f"  Cash on Hand:                  NIS {cash:>12,.2f}")
    print(f"  Open Positions (Market Value): NIS {positions_mkt_value:>12,.2f}")
    print(f"  TOTAL PORTFOLIO VALUE:         NIS {total_value:>12,.2f}")
    print(f"  {'─'*60}")
    print(f"  Total Return:                      {total_return:>+11.2f}%  (NIS {total_value-INITIAL_CASH:>+,.2f})")
    print(f"  {'─'*60}")
    print(f"  Realized Net P&L:              NIS {realized_net:>+12,.2f}")
    print(f"  Unrealized P&L (open pos):     NIS {unrealized:>+12,.2f}")
    print(f"  Combined P&L (Realized+Unreal):NIS {realized_net+unrealized:>+12,.2f}")
    print(f"  {'─'*60}")
    print(f"  Total Commissions Paid:        NIS {total_comm:>12,.2f}")
    print(f"  Total Capital Gains Tax Paid:  NIS {total_tax_paid:>12,.2f}")
    print(f"  {'─'*60}")
    print(f"  Total Buy Orders:              {len([t for t in trades if t.side=='BUY']):>14}")
    print(f"  Total Sell Orders:             {len(sell_all):>14}")
    print(f"  Closed Round-Trips:            {len(sell_all):>14}")
    print(f"  Win Rate (closed trades):      {win_rate:>13.0f}%")
    print(f"  Avg Win:                       NIS {avg_win:>+12,.2f}")
    print(f"  Avg Loss:                      NIS {avg_loss:>+12,.2f}")
    if wins and losses:
        rr = abs(avg_win / avg_loss)
        print(f"  Risk/Reward Ratio:             {rr:>14.2f}x")
    print(f"  Regime-blocked days (no buys): {regime_blocks:>14}")
    print(f"  Open Positions at End:         {len(positions):>14}")
    print(f"  {'='*W}\n")

    # ── Baseline comparison reminder ──────────────────────────────────────────
    print("  COMPARISON vs BASELINE (v1 — fixed 5% stop, no regime, no RS filter):")
    print(f"  v1:  +9.57%  |  Win Rate: 43%  |  63 round-trips  |  R/R: 1.99x")
    print(f"  v2:  {total_return:>+.2f}%  |  Win Rate: {win_rate:.0f}%  |  {len(sell_all)} round-trips  |  "
          + (f"R/R: {abs(avg_win/avg_loss):.2f}x" if wins and losses else "R/R: n/a"))
    print()


if __name__ == "__main__":
    run_backtest()
