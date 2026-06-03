"""
Live trading runner — connects to Interactive Brokers and runs
the ORB scalper on the configured TASE universe for one trading day.

Prerequisites:
  1. TWS or IB Gateway running on localhost (paper trading recommended)
  2. pip install ib_insync
  3. TASE market subscription active in IB account

Usage:
    cd trading-agent
    python -m scalper.run_live [--paper] [--tickers HAPO.TA LUMI.TA ...]
"""
from __future__ import annotations

import argparse
import logging
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def is_trading_day() -> bool:
    """TASE trades Sunday–Thursday."""
    return datetime.now().weekday() in (6, 0, 1, 2, 3)  # Sun=6, Mon=0, …, Thu=3


def main():
    parser = argparse.ArgumentParser(description="TASE ORB Scalper — live mode")
    parser.add_argument("--paper",   action="store_true", default=True,
                        help="Use paper trading (default=True)")
    parser.add_argument("--live",    action="store_true",
                        help="Use live trading account (REAL MONEY)")
    parser.add_argument("--port",    type=int, default=None,
                        help="IB port override (default: 7497 paper / 7496 live)")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Override ticker list (e.g. HAPO.TA LUMI.TA)")
    args = parser.parse_args()

    if args.live:
        logger.warning("=" * 60)
        logger.warning("  LIVE TRADING MODE — REAL MONEY AT RISK")
        logger.warning("  Press Ctrl+C within 10 seconds to abort.")
        logger.warning("=" * 60)
        import time
        time.sleep(10)
        paper = False
        port  = args.port or 7496
    else:
        paper = True
        port  = args.port or 7497

    if not is_trading_day():
        logger.error("Today is not a TASE trading day (Sun–Thu). Exiting.")
        sys.exit(1)

    try:
        from scalper.ib_interface import TASEBroker
        from scalper.config import SCALP_UNIVERSE, ib_cfg
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Run:  pip install ib_insync")
        sys.exit(1)

    cfg = ib_cfg
    if args.port:
        cfg.port = args.port

    tickers = args.tickers or SCALP_UNIVERSE[:5]   # default: top 5 by liquidity

    logger.info(f"Starting ORB scalper | paper={paper} | port={port}")
    logger.info(f"Tickers: {tickers}")

    broker = TASEBroker(cfg=cfg, paper=paper)

    try:
        broker.connect()
        logger.info(f"Portfolio value: ₪{broker.portfolio_value():,.0f}")
        logger.info(f"Cash available:  ₪{broker.cash_balance():,.0f}")
        broker.run_day(tickers)
    except KeyboardInterrupt:
        logger.info("Interrupted — closing all positions")
        broker.close_all()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        broker.close_all()
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
