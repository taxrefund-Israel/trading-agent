"""
Run TASE intraday VWAP scalper backtest.
Uses real 5-minute Yahoo Finance data (60-day window available for TASE).

Usage:
    cd trading-agent
    python -m intraday.run_backtest
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from intraday.vwap_scalper import (
    download_intraday, VWAPBacktest, print_results, plot_results,
    INTRADAY_UNIVERSE, INITIAL_CAPITAL,
)

SEP = "=" * 65


def main():
    print(SEP)
    print("  TASE INTRADAY VWAP SCALPER — BACKTEST")
    print("  Strategy: VWAP mean-reversion (long + short)")
    print("  Data: 5-minute bars, 60-day window, Yahoo Finance")
    print("  Broker model: Interactive Brokers (0.046%/side, min $2.50)")
    print(SEP)

    # Download 5-min bars
    print(f"\nDownloading 5-min bars for {len(INTRADAY_UNIVERSE)} stocks...")
    all_bars = download_intraday(INTRADAY_UNIVERSE, period="60d", interval="5m", verbose=True)

    if not all_bars:
        print("ERROR: no data downloaded. Check internet connection.")
        return

    print(f"\nLoaded {len(all_bars)} tickers.")

    total_bars = sum(len(df) for df in all_bars.values())
    print(f"Total 5-min bars: {total_bars:,}")

    # Run backtest
    print(f"\nRunning backtest on NIS {INITIAL_CAPITAL:,.0f} capital...")
    bt      = VWAPBacktest(capital=INITIAL_CAPITAL)
    results = bt.run(all_bars, verbose=True)

    if not results:
        print("No trades generated. Check signal parameters.")
        return

    # Print results
    print_results(results, note=f"Tickers: {len(all_bars)} | IB commission model")

    # Save chart
    out_png = Path(__file__).parent.parent / "intraday_vwap_backtest.png"
    plot_results(results, str(out_png))

    # Save trade log
    out_csv = Path(__file__).parent.parent / "intraday_trades.csv"
    results["trades_df"].to_csv(str(out_csv), index=False)
    print(f"Trade log saved -> {out_csv}")

    # Per-ticker breakdown
    df = results["trades_df"]
    print(f"\n  Per-ticker breakdown:")
    print(f"  {'Ticker':<12} {'Trades':>7} {'WinRate':>8} {'NetPnL':>10} {'AvgWin':>9} {'AvgLoss':>9}")
    print("  " + "-" * 58)
    for ticker, grp in df.groupby("ticker"):
        wins   = grp[grp["net_pnl"] > 0]
        losses = grp[grp["net_pnl"] <= 0]
        wr     = len(wins) / len(grp) * 100
        net    = grp["net_pnl"].sum()
        aw     = wins["net_pnl"].mean()  if len(wins)   else 0
        al     = losses["net_pnl"].mean() if len(losses) else 0
        print(f"  {ticker:<12} {len(grp):>7} {wr:>7.1f}% {net:>+10,.0f} {aw:>+9,.0f} {al:>+9,.0f}")


if __name__ == "__main__":
    main()
