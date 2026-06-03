"""
Entry point — run the 5-year ORB scalper backtest and display results.

Usage:
    cd trading-agent
    python -m scalper.run_backtest
"""
from __future__ import annotations

import sys
import os

# Allow running from the trading-agent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalper.backtest import run, plot_results

if __name__ == "__main__":
    print("TASE ORB Scalper -- 5-Year Backtest")
    print("Universe: TA-125 most liquid stocks")
    print("Capital:  NIS 100,000")
    print("Strategy: Opening Range Breakout (ORB) + RSI + Volume filter")
    print("Broker:   Interactive Brokers (TASE exchange)")
    print("-" * 60)

    results = run(verbose=True)

    if results:
        try:
            plot_results(results)
        except Exception as e:
            print(f"Chart error (non-fatal): {e}")

        # Save trade log to CSV
        trades_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "scalper_trades.csv"
        )
        results["trades_df"].to_csv(trades_path, index=False)
        print(f"\nTrade log saved -> scalper_trades.csv")
