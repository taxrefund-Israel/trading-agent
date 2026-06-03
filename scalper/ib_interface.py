"""
Interactive Brokers live trading interface for TASE scalping.

Requirements:
  - TWS (Trader Workstation) or IB Gateway running locally
  - Paper trading account recommended for testing (port 7497)
  - pip install ib_insync

TASE stocks in IB:
  - Exchange: "TASE"
  - Currency: "ILS"
  - SecType:  "STK"

Connection ports:
  7497 — TWS paper trading
  7496 — TWS live trading
  4002 — IB Gateway paper trading
  4001 — IB Gateway live trading
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Callable

logger = logging.getLogger(__name__)

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder, util
    from ib_insync import RealTimeBar, BarData
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logger.warning("ib_insync not installed. Run: pip install ib_insync")

from .config import IBConfig, ScalperParams, ib_cfg, params as P
from .strategy import LiveORBTracker, Signal


class TASEBroker:
    """
    Wrapper around ib_insync for TASE intraday scalping.

    Usage (paper trading):
        broker = TASEBroker(paper=True)
        broker.connect()
        broker.subscribe_bars("HAPO.TA", callback=my_callback)
        broker.place_order("HAPO.TA", "BUY", qty=100, limit_price=38.50)
        broker.disconnect()
    """

    def __init__(self, cfg: IBConfig = ib_cfg, paper: bool = True):
        if not IB_AVAILABLE:
            raise ImportError("Install ib_insync: pip install ib_insync")

        self.cfg     = cfg
        self.paper   = paper
        self.ib      = IB()
        self._contracts: dict[str, object] = {}   # ticker → IB Contract
        self._trackers: dict[str, LiveORBTracker] = {}
        self._bars:     dict[str, object] = {}    # ticker → ib_insync bars subscription

        # Active positions: ticker → {qty, entry, stop, target}
        self.positions: dict[str, dict] = {}
        self.order_ids: dict[str, int]  = {}

    # ─── Connection ────────────────────────────────────────────────────────────
    def connect(self):
        port = self.cfg.port   # 7497=TWS paper, 7496=TWS live
        logger.info(f"Connecting to IB {'(paper)' if self.paper else '(LIVE)'} "
                    f"at {self.cfg.host}:{port}")
        self.ib.connect(self.cfg.host, port, clientId=self.cfg.client_id,
                        timeout=self.cfg.timeout)
        logger.info(f"Connected. Server version: {self.ib.client.serverVersion()}")
        return self

    def disconnect(self):
        self.ib.disconnect()
        logger.info("Disconnected from IB.")

    # ─── Contract resolution ───────────────────────────────────────────────────
    def _contract(self, ticker: str) -> object:
        """Return a qualified IB Stock contract for a TASE ticker."""
        if ticker in self._contracts:
            return self._contracts[ticker]

        # Strip .TA suffix for IB symbol
        symbol = ticker.replace(".TA", "")
        contract = Stock(symbol, self.cfg.exchange, self.cfg.currency)
        details = self.ib.qualifyContracts(contract)
        if not details:
            raise ValueError(f"IB cannot find contract for {ticker} on TASE")
        self._contracts[ticker] = details[0]
        logger.info(f"Qualified contract: {ticker} → conId={details[0].conId}")
        return details[0]

    # ─── Account & portfolio ──────────────────────────────────────────────────
    def account_summary(self) -> dict:
        vals = self.ib.accountSummary()
        result = {}
        for v in vals:
            result[v.tag] = v.value
        return result

    def portfolio_value(self) -> float:
        summary = self.account_summary()
        return float(summary.get("NetLiquidation", 0))

    def cash_balance(self) -> float:
        summary = self.account_summary()
        return float(summary.get("AvailableFunds", 0))

    # ─── Real-time bars ────────────────────────────────────────────────────────
    def subscribe_bars(self, ticker: str,
                       callback: Callable[[str, object], None]) -> None:
        """
        Subscribe to 5-second real-time bars for a TASE stock.
        callback(ticker, bar) is called for each new bar.
        """
        contract = self._contract(ticker)
        tracker  = LiveORBTracker(ticker, P)
        self._trackers[ticker] = tracker

        def _on_bar(bars, has_new_bar):
            if not has_new_bar:
                return
            bar = bars[-1]
            now = datetime.now()
            in_orb = self._is_orb_period(now)
            sig = tracker.update(
                bar_time=bar.time,
                bar_open=bar.open,
                bar_high=bar.high,
                bar_low=bar.low,
                bar_close=bar.close,
                bar_vol=bar.volume,
                in_orb_period=in_orb,
            )
            callback(ticker, bar)
            if sig:
                logger.info(f"SIGNAL: {sig}")
                self._handle_signal(sig)

        bars_sub = self.ib.reqRealTimeBars(
            contract,
            barSize=5,          # 5-second bars
            whatToShow="TRADES",
            useRTH=True,        # regular trading hours only
        )
        bars_sub.updateEvent += _on_bar
        self._bars[ticker] = bars_sub

    def unsubscribe_bars(self, ticker: str):
        if ticker in self._bars:
            self.ib.cancelRealTimeBars(self._bars.pop(ticker))

    def _is_orb_period(self, now: datetime) -> bool:
        from .config import MARKET_OPEN_H, MARKET_OPEN_M, ORB_END_H, ORB_END_M
        t = now.time()
        from datetime import time as dtime
        open_t = dtime(MARKET_OPEN_H, MARKET_OPEN_M)
        end_t  = dtime(ORB_END_H,     ORB_END_M)
        return open_t <= t < end_t

    # ─── Order management ─────────────────────────────────────────────────────
    def place_bracket_order(self, ticker: str, qty: float,
                            entry: float, stop: float, target: float) -> list:
        """
        Submit a bracket order: limit entry + stop loss + limit take profit.
        Returns list of IB Trade objects.
        """
        contract  = self._contract(ticker)
        qty_int   = int(round(qty))

        parent = LimitOrder(
            action="BUY",
            totalQuantity=qty_int,
            lmtPrice=round(entry, 2),
            transmit=False,      # send as bracket group
            orderRef=f"ORB_{ticker}_{int(time.time())}",
        )
        stop_order = StopOrder(
            action="SELL",
            totalQuantity=qty_int,
            stopPrice=round(stop, 2),
            parentId=0,          # filled after IB assigns parent ID
            transmit=False,
        )
        take_profit = LimitOrder(
            action="SELL",
            totalQuantity=qty_int,
            lmtPrice=round(target, 2),
            parentId=0,
            transmit=True,       # transmit entire bracket together
        )

        trades = self.ib.bracketOrder(
            action="BUY",
            quantity=qty_int,
            limitPrice=round(entry, 2),
            takeProfitPrice=round(target, 2),
            stopLossPrice=round(stop, 2),
        )
        # Submit all three legs
        results = []
        for trade_def in trades:
            t = self.ib.placeOrder(contract, trade_def)
            results.append(t)
            logger.info(f"Order placed: {t.order.action} {qty_int} {ticker} "
                        f"@ {t.order.lmtPrice}")

        self.positions[ticker] = dict(
            qty=qty_int, entry=entry, stop=stop, target=target
        )
        return results

    def close_position(self, ticker: str) -> object | None:
        """Market-order close of an open position."""
        if ticker not in self.positions:
            logger.warning(f"No position in {ticker}")
            return None
        pos = self.positions.pop(ticker)
        contract = self._contract(ticker)
        order = MarketOrder("SELL", int(pos["qty"]))
        trade = self.ib.placeOrder(contract, order)
        logger.info(f"Closing {ticker}: SELL {pos['qty']} @ MARKET")
        return trade

    def close_all(self):
        """Emergency: close every open position at market."""
        for ticker in list(self.positions.keys()):
            self.close_position(ticker)

    # ─── Signal handler ───────────────────────────────────────────────────────
    def _handle_signal(self, sig: Signal):
        """Called when the live tracker fires a breakout signal."""
        if sig.ticker in self.positions:
            logger.info(f"Already in {sig.ticker} — ignoring new signal")
            return
        if len(self.positions) >= P.max_open_positions:
            logger.info("Max positions reached — ignoring signal")
            return

        cash = self.cash_balance()
        port = self.portfolio_value()

        # Risk-parity sizing
        risk_amount = port * P.risk_per_trade_pct
        stop_dist   = sig.entry - sig.stop
        if stop_dist <= 0:
            return
        qty = risk_amount / stop_dist
        max_qty = (port * P.max_position_pct) / sig.entry
        qty = min(qty, max_qty)
        qty = max(1, int(qty))

        if qty * sig.entry > cash * 0.95:
            logger.warning(f"Insufficient cash for {sig.ticker} — skipping")
            return

        logger.info(f"Placing bracket: {sig.ticker} qty={qty} "
                    f"entry={sig.entry} stop={sig.stop} target={sig.target}")
        self.place_bracket_order(sig.ticker, qty, sig.entry, sig.stop, sig.target)

    # ─── Intraday runner ──────────────────────────────────────────────────────
    def run_day(self, tickers: list[str]):
        """
        Run a full trading day:
          1. Subscribe to real-time bars for each ticker.
          2. Process events until 16:30 (close cutoff).
          3. Close all remaining positions.
          4. Unsubscribe.
        """
        from .config import CLOSE_CUTOFF_H, CLOSE_CUTOFF_M
        from datetime import time as dtime

        cutoff = dtime(CLOSE_CUTOFF_H, CLOSE_CUTOFF_M)

        def noop_callback(ticker, bar):
            pass   # actual logic is inside subscribe_bars / _handle_signal

        for ticker in tickers:
            self.subscribe_bars(ticker, callback=noop_callback)

        logger.info(f"Trading day started — monitoring {tickers}")

        try:
            while True:
                self.ib.sleep(1)   # yield to event loop
                if datetime.now().time() >= cutoff:
                    logger.info("16:30 cutoff — closing all positions")
                    self.close_all()
                    break
        finally:
            for ticker in tickers:
                self.unsubscribe_bars(ticker)
            logger.info("Trading day finished.")


# ─── Meitav Dash placeholder ──────────────────────────────────────────────────
class MeitavDashBroker:
    """
    Stub for Meitav Dash REST API integration.

    Meitav Dash (מיטב דש) offers a REST API for retail clients at:
      https://www.meitavdash.co.il/trading/api

    Authentication uses OAuth2 with personal API key.
    This stub shows the interface contract; fill in the HTTP calls
    once you have API credentials from Meitav Dash.
    """

    BASE_URL = "https://openapi.meitavdash.co.il"  # example — verify with Meitav

    def __init__(self, api_key: str, account_id: str, paper: bool = True):
        self.api_key    = api_key
        self.account_id = account_id
        self.paper      = paper
        self.positions: dict[str, dict] = {}
        logger.info("MeitavDashBroker initialized (stub)")

    def connect(self) -> bool:
        # TODO: POST /auth/token with api_key
        raise NotImplementedError(
            "MeitavDash API integration is a stub. "
            "Obtain your API key from Meitav Dash and implement OAuth2 flow."
        )

    def place_order(self, ticker: str, side: str, qty: int,
                    price: float | None = None) -> dict:
        # TODO: POST /orders with symbol, side, quantity, price
        raise NotImplementedError

    def get_portfolio(self) -> dict:
        # TODO: GET /accounts/{account_id}/portfolio
        raise NotImplementedError

    def close_position(self, ticker: str) -> dict:
        raise NotImplementedError
