"""
ORB (Opening Range Breakout) + VWAP intraday strategy for TASE.

Two modes:
  1. Backtest mode  — works on daily OHLCV bars, simulates intraday fills.
  2. Live mode      — receives real-time 5-sec bars, emits BUY/SELL signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal

from .config import ScalperParams


# ─── Signal dataclass ─────────────────────────────────────────────────────────
@dataclass
class Signal:
    ticker:     str
    direction:  Literal["long", "flat"]
    entry:      float
    stop:       float
    target:     float
    confidence: float    # 0–1 composite score
    reason:     str = ""


# ─── Indicator helpers ────────────────────────────────────────────────────────
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical = (high + low + close) / 3
    cum_tp_vol = (typical * volume).cumsum()
    cum_vol    = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


# ─── Pre-trade filter (uses multi-day history) ────────────────────────────────
def compute_indicators(df: pd.DataFrame, p: ScalperParams) -> pd.DataFrame:
    """
    Add indicator columns to a daily OHLCV DataFrame.
    Call once per ticker, then slice row by row in the backtest loop.
    """
    df = df.copy()
    df["sma50"]  = df["Close"].rolling(50).mean()
    df["sma200"] = df["Close"].rolling(200).mean()
    df["rsi"]    = _rsi(df["Close"], p.rsi_period)
    df["atr"]    = _atr(df["High"], df["Low"], df["Close"], p.atr_period)
    df["vol20"]  = df["Volume"].rolling(20).mean()

    # Relative strength vs index (set externally via merge — skipped here)
    df["trend_bull"] = df["Close"] > df["sma50"]

    return df


# ─── Daily-bar ORB signal generator ──────────────────────────────────────────
def daily_orb_signal(row: pd.Series, p: ScalperParams,
                     index_above_sma: bool = True) -> Signal | None:
    """
    Generate a momentum breakout signal from a single daily OHLCV row.

    Uses the PREVIOUS day's data (passed in `row`) to generate a signal
    for NEXT day's open.  The backtest loop calls this with row[t-1] and
    executes the trade on day t.

    Signal conditions (checked on the signal bar, i.e. previous day):
      1. Stock is above its SMA-50  (trend filter)
      2. RSI > rsi_min_bull         (momentum)
      3. Volume >= min_volume_ratio × 20d avg  (volume confirmation)
      4. Index is above its SMA-50  (market regime — passed in)
      5. Close > previous close (momentum continuation)

    Entry next day:
      entry  = prev_close × (1 + orb_breakout_pct)   # slight premium
      stop   = entry × (1 - orb_stop_pct)              # tight stop below entry
      target = entry + risk × target_risk_ratio
    """
    if pd.isna(row.get("atr")) or pd.isna(row.get("rsi")) or pd.isna(row.get("sma50")):
        return None

    # Market regime: only long when index above SMA-50
    if not index_above_sma:
        return None

    # Stock trend filter — above SMA-50
    if not row["trend_bull"]:
        return None

    # RSI momentum filter
    if row["rsi"] < p.rsi_min_bull:
        return None

    # Volume filter
    vol20 = row.get("vol20", float("nan"))
    if pd.notna(vol20) and vol20 > 0 and row["Volume"] < vol20 * p.min_volume_ratio:
        return None

    prev_close = row["Close"]
    entry  = prev_close * (1 + p.orb_breakout_pct)
    stop   = entry * (1 - p.orb_stop_pct)
    risk   = entry - stop
    target = entry + risk * p.target_risk_ratio

    # Confidence score
    rsi_score  = min((row["rsi"] - p.rsi_min_bull) / (70 - p.rsi_min_bull), 1.0)
    vol_ratio  = (row["Volume"] / vol20) if (pd.notna(vol20) and vol20 > 0) else 1.0
    vol_score  = min(vol_ratio / 2.0, 1.0)
    confidence = (rsi_score + vol_score) / 2

    return Signal(
        ticker=str(row.get("ticker", "?")),
        direction="long",
        entry=round(entry, 4),
        stop=round(stop, 4),
        target=round(target, 4),
        confidence=round(confidence, 3),
        reason=f"MomBreakout | RSI={row['rsi']:.1f} | vol_ratio={vol_ratio:.2f}",
    )


# ─── Simulate trade outcome on a daily bar ────────────────────────────────────
def simulate_fill(sig: Signal, row: pd.Series) -> tuple[float, str]:
    """
    Given a signal and the same day's OHLCV bar, return (pnl_pct, outcome).

    Conservative rules:
      - If high < entry           → no fill (price never reached entry)
      - If both stop AND target hit → assume stop hit first (worst-case)
      - If target hit first       → profit = (target - entry) / entry
      - If stop hit               → loss   = (stop  - entry) / entry  (negative)
      - Else                      → exit at close
    """
    high  = row["High"]
    low   = row["Low"]
    close = row["Close"]

    entry  = sig.entry
    stop   = sig.stop
    target = sig.target

    # No fill: price never reached entry
    if high < entry:
        return 0.0, "no_fill"

    # Both levels hit — assume stop first (conservative)
    if low <= stop and high >= target:
        pct = (stop - entry) / entry
        return pct, "stop_then_target"

    if high >= target:
        pct = (target - entry) / entry
        return pct, "target"

    if low <= stop:
        pct = (stop - entry) / entry
        return pct, "stop"

    # Exit at close (EOD flat)
    pct = (close - entry) / entry
    return pct, "eod"


# ─── Live signal generator (called per real-time bar update) ─────────────────
class LiveORBTracker:
    """
    Tracks the opening range in real time.
    Feed it 5-second bars as they arrive; call signal() after ORB period ends.
    """

    def __init__(self, ticker: str, p: ScalperParams):
        self.ticker = ticker
        self.p = p
        self._bars: list[dict] = []
        self.orb_high: float | None = None
        self.orb_low:  float | None = None
        self.orb_locked = False
        self.entry_fired = False
        self.vwap_num: float = 0.0   # ΣP×V (cumulative)
        self.vwap_den: float = 0.0   # ΣV

    def update(self, bar_time, bar_open, bar_high, bar_low, bar_close, bar_vol,
               in_orb_period: bool) -> Signal | None:
        """Feed one real-time bar. Returns a Signal when breakout is confirmed."""

        # Update rolling VWAP
        typical = (bar_high + bar_low + bar_close) / 3
        self.vwap_num += typical * bar_vol
        self.vwap_den += bar_vol
        vwap = self.vwap_num / self.vwap_den if self.vwap_den > 0 else bar_close

        if in_orb_period:
            # Accumulate opening range
            if self.orb_high is None:
                self.orb_high = bar_high
                self.orb_low  = bar_low
            else:
                self.orb_high = max(self.orb_high, bar_high)
                self.orb_low  = min(self.orb_low,  bar_low)
            return None

        # ORB period just ended — lock the range
        if not self.orb_locked and self.orb_high is not None:
            self.orb_locked = True

        if not self.orb_locked or self.entry_fired:
            return None

        # Long breakout: price closes above OR high AND above VWAP
        if bar_close > self.orb_high and bar_close > vwap:
            entry  = bar_close * (1 + self.p.orb_breakout_pct)
            stop   = self.orb_low
            risk   = entry - stop
            if risk <= 0:
                return None
            target = entry + risk * self.p.target_risk_ratio
            self.entry_fired = True
            return Signal(
                ticker=self.ticker,
                direction="long",
                entry=round(entry, 4),
                stop=round(stop, 4),
                target=round(target, 4),
                confidence=0.75,
                reason=f"Live ORB breakout | VWAP={vwap:.2f} | OR={self.orb_low:.2f}-{self.orb_high:.2f}",
            )

        return None
