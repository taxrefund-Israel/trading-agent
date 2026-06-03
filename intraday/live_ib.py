"""
TASE Intraday VWAP Scalper — Live IB Trading
=============================================
Connects to Interactive Brokers TWS via ib_insync.
Uses 5-second real-time bars to execute VWAP mean-reversion trades.

Prerequisites:
  - TWS (Thinkorswim/IB workstation) running on localhost
  - Paper trading account (port 7497) or live (7496)
  - pip install ib_insync

Usage:
    # Paper trading
    python -m intraday.live_ib --paper

    # Live (real money — be careful!)
    python -m intraday.live_ib --live

    # Custom tickers
    python -m intraday.live_ib --paper --tickers POLI.TA LUMI.TA MZTF.TA
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from intraday.vwap_scalper import (
    INTRADAY_UNIVERSE, INITIAL_CAPITAL, MAX_POSITIONS, POSITION_SIZE,
    ENTRY_BAND, TARGET_BAND, STOP_PCT, TIME_LIMIT, VOLUME_MULT,
    COMMISSION_PCT, COMMISSION_MIN, SLIPPAGE_PCT,
    detect_signal, OpenPosition,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vwap_live")

IL_TZ = ZoneInfo("Asia/Jerusalem")
MARKET_OPEN  = (9, 59)    # H, M
EOD_CUTOFF   = (16, 30)   # close all positions by 16:30 IL
MARKET_CLOSE = (17, 14)


def now_il():
    return datetime.now(IL_TZ)


class VWAPState:
    """Per-ticker rolling VWAP state, reset each morning."""
    def __init__(self, ticker: str):
        self.ticker  = ticker
        self.cum_pv  = 0.0   # price × volume cumsum
        self.cum_vol = 0.0   # volume cumsum
        self._date   = None
        self.bars: deque = deque(maxlen=30)   # recent bars for vol_ma

    def update(self, bar) -> float:
        """Feed one 5-second bar, return current VWAP."""
        date = bar.date.date() if hasattr(bar.date, "date") else bar.date
        if date != self._date:
            self.cum_pv  = 0.0
            self.cum_vol = 0.0
            self._date   = date

        typical = (bar.high + bar.low + bar.close) / 3
        self.cum_pv  += typical * bar.volume
        self.cum_vol += bar.volume
        self.bars.append({
            "Close":  bar.close,
            "Volume": bar.volume,
            "High":   bar.high,
            "Low":    bar.low,
        })
        return self.cum_pv / self.cum_vol if self.cum_vol else np.nan

    @property
    def vol_ma(self) -> float:
        if len(self.bars) < 5:
            return 0.0
        vols = [b["Volume"] for b in self.bars]
        return float(np.mean(vols[-20:]))

    def last_bars_series(self):
        bars = list(self.bars)
        if len(bars) < 3:
            return None, None, None
        return (
            pd.Series(bars[-1]),
            pd.Series(bars[-2]),
            pd.Series(bars[-3]),
        )


class IBLiveTrader:
    """Live TASE VWAP scalper using ib_insync."""

    def __init__(self, tickers: list[str], capital: float,
                 port: int = 7497, client_id: int = 11):
        self.tickers   = tickers
        self.capital   = capital
        self.port      = port
        self.client_id = client_id
        self.ib        = None

        self.positions: dict[str, OpenPosition]  = {}
        self.vwap_state: dict[str, VWAPState]    = {}
        self.daily_trades: list[dict]             = []
        self._bar_count: dict[str, int]           = defaultdict(int)
        self._subscriptions: dict[str, object]    = {}   # ticker -> bar subscription

    def connect(self):
        try:
            from ib_insync import IB
        except ImportError:
            raise RuntimeError("ib_insync not installed. Run: pip install ib_insync")

        self.ib = IB()
        log.info(f"Connecting to IB TWS on port {self.port} (client_id={self.client_id})")
        self.ib.connect("127.0.0.1", self.port, clientId=self.client_id, timeout=20)
        log.info("Connected.")

    def _contract(self, ticker: str):
        from ib_insync import Stock
        symbol = ticker.replace(".TA", "")
        return Stock(symbol, "TASE", "ILS")

    def _place_bracket(self, ticker: str, direction: str,
                       qty: float, entry: float,
                       stop: float, target: float) -> list:
        from ib_insync import MarketOrder, StopOrder, LimitOrder, BracketOrder

        action = "BUY" if direction == "long" else "SELL"
        rev    = "SELL" if direction == "long" else "BUY"

        parent  = MarketOrder(action, qty)
        sl      = StopOrder(rev, qty, stop)
        tp      = LimitOrder(rev, qty, target)
        sl.parentId  = parent.orderId
        tp.parentId  = parent.orderId
        sl.transmit = False
        tp.transmit = True

        contract = self._contract(ticker)
        trades   = self.ib.placeOrder(contract, parent)
        self.ib.placeOrder(contract, sl)
        self.ib.placeOrder(contract, tp)
        log.info(
            f"OPEN {direction.upper()} {ticker}  qty={qty:.0f}  "
            f"entry~{entry:.3f}  stop={stop:.3f}  target={target:.3f}"
        )
        return trades

    def _close_market(self, ticker: str, direction: str, qty: float, reason: str):
        from ib_insync import MarketOrder
        action   = "SELL" if direction == "long" else "BUY"
        contract = self._contract(ticker)
        self.ib.placeOrder(contract, MarketOrder(action, qty))
        log.info(f"CLOSE {ticker} ({reason})")

    def _on_bar(self, bars, ticker: str):
        """Called each time a new 5-second bar arrives."""
        if not bars:
            return
        bar = bars[-1]
        ts  = now_il()

        # Update VWAP
        state = self.vwap_state[ticker]
        vwap  = state.update(bar)
        self._bar_count[ticker] += 1

        # ── Manage open position ──────────────────────────────────────────────
        if ticker in self.positions:
            pos = self.positions[ticker]
            bars_held = self._bar_count[ticker] - pos.entry_bar

            if pos.direction == "long":
                if bar.low <= pos.stop_price:
                    self._close_market(ticker, pos.direction, pos.qty, "stop")
                    del self.positions[ticker]
                    return
                if bar.high >= pos.target_price and bars_held >= 1:
                    self._close_market(ticker, pos.direction, pos.qty, "target")
                    del self.positions[ticker]
                    return
            else:
                if bar.high >= pos.stop_price:
                    self._close_market(ticker, pos.direction, pos.qty, "stop")
                    del self.positions[ticker]
                    return
                if bar.low <= pos.target_price and bars_held >= 1:
                    self._close_market(ticker, pos.direction, pos.qty, "target")
                    del self.positions[ticker]
                    return

            if bars_held >= TIME_LIMIT * 12:   # 5-sec bars: 6 bars × 12 = 5-min × 6
                self._close_market(ticker, pos.direction, pos.qty, "time")
                del self.positions[ticker]
                return

        # ── Check new signal ──────────────────────────────────────────────────
        if len(self.positions) >= MAX_POSITIONS or ticker in self.positions:
            return

        t  = ts.hour * 60 + ts.minute
        open_min  = MARKET_OPEN[0]  * 60 + MARKET_OPEN[1]
        cutoff_min= EOD_CUTOFF[0]   * 60 + EOD_CUTOFF[1]
        if t < open_min + 60 or t >= cutoff_min:   # skip first hour + EOD
            return

        b0, b1, b2 = state.last_bars_series()
        if b0 is None:
            return

        sig = detect_signal(b0, b1, b2, vwap, state.vol_ma)
        if sig is None:
            return

        price    = bar.close
        pos_val  = self.capital * POSITION_SIZE
        qty      = int(pos_val / price)
        if qty < 1:
            return

        if sig == "long":
            stop_p   = price * (1 - STOP_PCT)
            target_p = vwap  * (1 - TARGET_BAND)
        else:
            stop_p   = price * (1 + STOP_PCT)
            target_p = vwap  * (1 + TARGET_BAND)

        self._place_bracket(ticker, sig, qty, price, stop_p, target_p)
        self.positions[ticker] = OpenPosition(
            ticker=ticker, direction=sig,
            entry_time=ts, entry_bar=self._bar_count[ticker],
            entry_price=price, qty=qty, position_val=pos_val,
            stop_price=stop_p, target_price=target_p,
            vwap_at_entry=vwap, deviation_pct=((price - vwap) / vwap * 100),
        )

    def subscribe_all(self):
        from ib_insync import RealTimeBarList
        for ticker in self.tickers:
            self.vwap_state[ticker] = VWAPState(ticker)
            contract = self._contract(ticker)
            bars_obj = self.ib.reqRealTimeBars(
                contract, barSize=5, whatToShow="TRADES", useRTH=True
            )
            bars_obj.updateEvent += (
                lambda bars, hasNew, t=ticker: self._on_bar(bars, t)
            )
            self._subscriptions[ticker] = bars_obj
            log.info(f"Subscribed: {ticker}")

    def close_all_eod(self):
        """Emergency close all open positions at market."""
        for ticker, pos in list(self.positions.items()):
            self._close_market(ticker, pos.direction, pos.qty, "eod")
        self.positions.clear()
        log.info("All positions closed (EOD).")

    def run(self):
        self.connect()
        self.subscribe_all()

        log.info("Live loop started. Press Ctrl-C to stop.")
        try:
            while True:
                self.ib.sleep(1)
                t = now_il()
                cutoff = t.replace(
                    hour=EOD_CUTOFF[0], minute=EOD_CUTOFF[1],
                    second=0, microsecond=0,
                )
                if t >= cutoff and self.positions:
                    log.info("EOD cutoff reached — closing all positions.")
                    self.close_all_eod()
                    break
        except KeyboardInterrupt:
            log.info("Interrupted by user.")
            self.close_all_eod()
        finally:
            if self.ib and self.ib.isConnected():
                self.ib.disconnect()
                log.info("Disconnected from IB.")


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TASE Intraday VWAP Scalper — Live Trading via IB"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--paper", action="store_true",
                     help="Connect to TWS paper account (port 7497)")
    grp.add_argument("--live", action="store_true",
                     help="Connect to TWS live account (port 7496) — REAL MONEY")
    parser.add_argument("--tickers", nargs="*", default=INTRADAY_UNIVERSE,
                        help="Space-separated tickers (default: built-in universe)")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL,
                        help="Trading capital in NIS (default: 100000)")
    args = parser.parse_args()

    port = 7497 if args.paper else 7496
    mode = "PAPER" if args.paper else "LIVE (REAL MONEY)"

    print(f"\nMode:    {mode}")
    print(f"Port:    {port}")
    print(f"Capital: NIS {args.capital:,.0f}")
    print(f"Tickers: {args.tickers}")

    if args.live:
        confirm = input("\nWARNING: Live trading with real money. Type YES to continue: ")
        if confirm.strip().upper() != "YES":
            print("Aborted.")
            return

    trader = IBLiveTrader(
        tickers=args.tickers,
        capital=args.capital,
        port=port,
    )
    trader.run()


if __name__ == "__main__":
    main()
