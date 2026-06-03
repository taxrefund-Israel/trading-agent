"""
Run TASE ORB intraday backtest.

Usage:
    cd trading-agent
    python -m intraday.run_orb
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from intraday.orb_scalper import (
    download_intraday, ORBBacktest, print_results, plot_results,
    INTRADAY_UNIVERSE, INITIAL_CAPITAL,
    OR_BARS, BREAKOUT_PCT, TARGET_MULT, POSITION_SIZE, MAX_POSITIONS,
    VOLUME_MULT, LONG_ONLY,
)

SEP = "=" * 65


def main():
    print(SEP)
    print("  TASE ORB INTRADAY SCALPER — BACKTEST")
    print("  Strategy: Opening Range Breakout (first 30 min)")
    print("  Data: 5-minute bars, 60-day window")
    print("  Broker: Interactive Brokers (0.046%/side, min $2.50)")
    print(SEP)
    print(f"\n  Parameters:")
    print(f"    Opening range:  first {OR_BARS} bars (30 min, 09:59-10:29 IL)")
    print(f"    Breakout buffer:{BREAKOUT_PCT*100:.1f}% above/below OR range")
    print(f"    Target:         {TARGET_MULT}x opening range height")
    print(f"    Stop:           opposite edge of opening range")
    print(f"    Max positions:  {MAX_POSITIONS}  |  Position size: {POSITION_SIZE*100:.0f}%")
    print(f"    Volume filter:  {VOLUME_MULT}x OR avg volume")
    print(f"    Long only:      {LONG_ONLY}")

    print(f"\nDownloading 5-min bars for {len(INTRADAY_UNIVERSE)} stocks...")
    all_bars = download_intraday(INTRADAY_UNIVERSE, period="60d", interval="5m", verbose=True)

    if not all_bars:
        print("ERROR: no data downloaded.")
        return

    total = sum(len(df) for df in all_bars.values())
    print(f"\nLoaded {len(all_bars)} tickers, {total:,} total bars.")

    # Download TA-125 daily bars for regime filter
    print("\nDownloading TA-125 daily data for regime filter...")
    import yfinance as yf
    idx_df = yf.download("^TA125.TA", period="90d", interval="1d",
                         progress=False, multi_level_index=False, auto_adjust=True)
    idx_df.columns = [c.capitalize() for c in idx_df.columns]
    if idx_df.index.tzinfo is not None:
        idx_df.index = idx_df.index.tz_localize(None)
    print(f"  Index: {len(idx_df)} daily bars")

    print(f"\nRunning ORB backtest on NIS {INITIAL_CAPITAL:,.0f}...")
    bt      = ORBBacktest(capital=INITIAL_CAPITAL)
    results = bt.run(all_bars, index_daily=idx_df, verbose=True)

    if not results:
        print("No trades generated. Tighten breakout filter or add tickers.")
        return

    print_results(results, note=f"10 tickers | IB commission | {results['backtest_days']}d window")

    # Chart
    out_png = Path(__file__).parent.parent / "orb_backtest.png"
    plot_results(results, str(out_png))

    # Trade log
    out_csv = Path(__file__).parent.parent / "orb_trades.csv"
    results["trades_df"].to_csv(str(out_csv), index=False)
    print(f"Trade log saved -> {out_csv}")

    # Per-ticker summary
    df = results["trades_df"]
    print(f"\n  Per-ticker breakdown:")
    print(f"  {'Ticker':<12} {'Trades':>7} {'WinRate':>8} {'NetPnL':>10} {'AvgOR%':>8} {'Outcome'}")
    print("  " + "-" * 65)
    for ticker, grp in df.groupby("ticker"):
        wins = grp[grp["net_pnl"] > 0]
        wr   = len(wins) / len(grp) * 100
        net  = grp["net_pnl"].sum()
        avg_or = grp["or_height"].mean() / grp["entry_price"].mean() * 100
        oc   = grp["outcome"].value_counts().to_dict()
        oc_s = " ".join(f"{k[0]}:{v}" for k, v in sorted(oc.items()))
        print(f"  {ticker:<12} {len(grp):>7} {wr:>7.1f}% {net:>+10,.0f} {avg_or:>7.2f}%  {oc_s}")

    # OR height statistics
    avg_or_pct = (df["or_height"] / df["entry_price"] * 100).mean()
    print(f"\n  Avg OR height: {avg_or_pct:.3f}% of price")
    print(f"  Avg target dist: {avg_or_pct * TARGET_MULT:.3f}%  "
          f"Avg stop dist (approx): {avg_or_pct * 0.5:.3f}%")


if __name__ == "__main__":
    main()
