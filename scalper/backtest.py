"""
5-year intraday ORB backtest for TASE using daily OHLCV data.

Simulation model (intraday — all trades open and close SAME DAY):
  Entry:  today's open + orb_breakout_pct (0.2%)  — simulates the ORB trigger
  Stop:   entry - orb_stop_pct (0.4%)              — below opening support
  Target: entry + risk × target_risk_ratio (0.8%)  — 2× risk reward

Fill simulation on today's OHLCV bar:
  - If high < entry                     → no trade (breakout not triggered)
  - If both stop and target hit:
      use close as tie-breaker:
        close > entry × 1.004 (upper half)  → target hit first (profit)
        else                                 → stop hit first (loss)
  - If only target hit                  → profit
  - If only stop hit                    → loss
  - Else (neither)                      → exit at close

Signal filter (previous day):
  - RSI(14) > 42       (mild upward momentum)
  - Price above SMA-20 (short-term trend)
  - Volume >= 80% of 20-day avg (liquidity confirmation)
  - Index above SMA-50 (market regime — no trading in bear markets)
"""
from __future__ import annotations

import warnings
from typing import NamedTuple
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from .config import (
    SCALP_UNIVERSE, INDEX_TICKER,
    INITIAL_CASH, SIM_START, SIM_END, FETCH_START, WARMUP_DAYS,
    params as P,
)
from .strategy import compute_indicators


# ─── Trade result ─────────────────────────────────────────────────────────────
class TradeResult(NamedTuple):
    date:       pd.Timestamp
    ticker:     str
    open_p:     float
    entry:      float
    stop:       float
    target:     float
    exit_price: float
    gross_pct:  float      # return on entry price
    net_pnl:    float      # after commission + slippage + tax
    outcome:    str
    portfolio:  float


# ─── Simulate a single intraday trade on a daily OHLCV bar ───────────────────
def simulate_trade(
    today: pd.Series,
    entry: float, stop: float, target: float,
) -> tuple[float, float, str]:
    """
    Returns (gross_pct, exit_price, outcome).
    gross_pct is the gain/loss as a fraction of entry price.
    """
    high  = today["High"]
    low   = today["Low"]
    close = today["Close"]

    # Entry not triggered — price never rallied to entry level
    if high < entry:
        return 0.0, entry, "no_fill"

    # Both stop and target hit on same bar — use close as tie-breaker
    if low <= stop and high >= target:
        upper_half = close >= (stop + target) / 2
        if upper_half:
            pct = (target - entry) / entry
            return pct, target, "target"
        else:
            pct = (stop - entry) / entry
            return pct, stop, "stop"

    if high >= target:
        pct = (target - entry) / entry
        return pct, target, "target"

    if low <= stop:
        pct = (stop - entry) / entry
        return pct, stop, "stop"

    # Exit at close (end-of-day flat)
    pct = (close - entry) / entry
    return pct, close, "eod"


# ─── Data fetching ────────────────────────────────────────────────────────────
def _fetch(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    print(f"Fetching {len(tickers)} TASE tickers from Yahoo Finance...")
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False,
                             multi_level_index=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Normalize column names
            df.columns = [c.capitalize() if c.lower() in
                          ("open","high","low","close","volume") else c
                          for c in df.columns]
            df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if len(df) < 60:
                print(f"  {ticker}: only {len(df)} days -- skipped")
                continue
            df["ticker"] = ticker
            data[ticker] = df
            print(f"  {ticker}: {len(df)} days ok")
        except Exception as e:
            print(f"  {ticker}: FAILED -- {e}")
    return data


# ─── Signal filter (applied to PREVIOUS day's row) ───────────────────────────
def _has_signal(prev: pd.Series, index_bull: bool) -> bool:
    """True if prev day's indicators pass the entry filter."""
    if not index_bull:
        return False
    if pd.isna(prev.get("rsi")) or pd.isna(prev.get("sma20")):
        return False
    if prev["rsi"] < 42:
        return False
    if prev["Close"] < prev.get("sma20", 0):
        return False
    vol20 = prev.get("vol20", float("nan"))
    if pd.notna(vol20) and vol20 > 0 and prev["Volume"] < vol20 * 0.80:
        return False
    return True


def compute_indicators_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Lighter indicator set optimized for the scalper filter."""
    df = df.copy()
    close = df["Close"]
    df["sma20"]  = close.rolling(20).mean()
    df["sma50"]  = close.rolling(50).mean()
    df["sma200"] = close.rolling(200).mean()
    df["vol20"]  = df["Volume"].rolling(20).mean()

    # RSI via exponential MA
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    return df


# ─── Main backtest ────────────────────────────────────────────────────────────
def run(verbose: bool = True) -> dict:
    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    all_data = _fetch(SCALP_UNIVERSE, start=FETCH_START, end=SIM_END)
    if not all_data:
        raise RuntimeError("No data downloaded — check internet connection.")

    print(f"Fetching index {INDEX_TICKER}...")
    try:
        idx_raw = yf.download(INDEX_TICKER, start=FETCH_START, end=SIM_END,
                              auto_adjust=True, progress=False,
                              multi_level_index=False)
        if isinstance(idx_raw.columns, pd.MultiIndex):
            idx_raw.columns = idx_raw.columns.get_level_values(0)
        idx_raw.columns = [c.capitalize() if c.lower() in
                           ("open","high","low","close","volume") else c
                           for c in idx_raw.columns]
        idx_raw["sma50"]    = idx_raw["Close"].rolling(50).mean()
        idx_raw["idx_bull"] = idx_raw["Close"] > idx_raw["sma50"]
        print(f"  {INDEX_TICKER}: {len(idx_raw)} days ok")
        has_index = True
    except Exception as e:
        print(f"  Index failed: {e} -- regime filter OFF")
        idx_raw = None
        has_index = False

    # ── 2. Compute indicators ─────────────────────────────────────────────────
    for ticker in all_data:
        all_data[ticker] = compute_indicators_v2(all_data[ticker])

    # ── 3. Build sorted date list ─────────────────────────────────────────────
    all_dates = sorted({d for df in all_data.values() for d in df.index})
    sim_start = pd.Timestamp(SIM_START)

    # ── 4. Portfolio state ────────────────────────────────────────────────────
    portfolio   = INITIAL_CASH
    peak        = INITIAL_CASH
    trades:    list[TradeResult] = []
    daily_snap: list[dict]       = []
    prev_date:  pd.Timestamp | None = None
    # Annual tax tracking (Israeli capital gains: 25% on NET annual profits)
    year_start_portfolio = INITIAL_CASH
    current_year = pd.Timestamp(SIM_START).year

    for i, dt in enumerate(all_dates):
        if prev_date is None:
            prev_date = dt
            continue

        # Guardrail: stop if portfolio falls >15% below peak (only after warmup)
        drawdown = (peak - portfolio) / peak
        if dt >= sim_start and drawdown >= 0.15:
            if verbose:
                print(f"[{dt.date()}] GUARDRAIL: drawdown >= 15%, halting.")
            break

        # Market regime
        index_bull = True
        if has_index and dt in idx_raw.index:
            index_bull = bool(idx_raw.loc[dt, "idx_bull"])

        # Skip warmup period (don't trade, just accumulate indicator history)
        if dt < sim_start:
            prev_date = dt
            continue

        # ── 5. Generate signals, rank, execute top ones ───────────────────────
        candidates = []
        for ticker, df in all_data.items():
            if prev_date not in df.index or dt not in df.index:
                continue

            prev_row = df.loc[prev_date]
            today    = df.loc[dt]

            # Signal filter (prev day fundamentals)
            if not _has_signal(prev_row, index_bull):
                continue

            # Gap-up filter: today must open above previous close (momentum)
            # Key ORB insight: gap-up days have much higher win rates
            today_open = today["Open"]
            if today_open < prev_row["Close"] * 0.999:
                continue   # gapping down — skip ORB

            # Momentum quality score: RSI + volume + gap size
            vol20 = prev_row.get("vol20", float("nan"))
            vol_ratio = (prev_row["Volume"] / vol20
                         ) if (pd.notna(vol20) and vol20 > 0) else 1.0
            gap_pct = (today_open / prev_row["Close"] - 1) * 100
            rsi_val = prev_row.get("rsi", 50)
            score   = (rsi_val - 40) * 0.5 + vol_ratio * 2.0 + gap_pct * 3.0

            # Entry levels based on TODAY's open
            entry  = today_open * (1 + P.orb_breakout_pct)   # 0.2% above open
            stop   = entry       * (1 - P.orb_stop_pct)        # 0.4% below entry
            risk   = entry - stop
            target = entry + risk * P.target_risk_ratio         # 1.2% above entry

            candidates.append((score, ticker, today, entry, stop, target))

        # Sort by quality score, take top signals only
        candidates.sort(key=lambda x: -x[0])
        day_trades = 0
        for score, ticker, today, entry, stop, target in candidates:
            if day_trades >= P.max_daily_trades:
                break

            gross_pct, exit_price, outcome = simulate_trade(today, entry, stop, target)
            if outcome == "no_fill":
                continue

            # Transaction costs
            position_value = (portfolio * P.risk_per_trade_pct / P.orb_stop_pct)
            position_value = min(position_value, portfolio * P.max_position_pct)
            commission = position_value * P.commission_pct * 2
            slippage   = position_value * P.slippage_pct
            gross_pnl  = position_value * gross_pct
            net_pnl    = gross_pnl - commission - slippage

            portfolio += net_pnl
            peak = max(peak, portfolio)
            day_trades += 1

            trades.append(TradeResult(
                date=dt, ticker=ticker,
                open_p=today["Open"],
                entry=round(entry, 4),
                stop=round(stop, 4),
                target=round(target, 4),
                exit_price=round(exit_price, 4),
                gross_pct=round(gross_pct * 100, 3),
                net_pnl=round(net_pnl, 2),
                outcome=outcome,
                portfolio=round(portfolio, 2),
            ))

        # Year-end tax deduction (25% on annual net gain)
        if dt.year != current_year:
            annual_gain = portfolio - year_start_portfolio
            if annual_gain > 0:
                tax = annual_gain * P.tax_rate
                portfolio -= tax
                if verbose:
                    print(f"  [{dt.year-1} tax] Annual gain: NIS {annual_gain:,.0f} "
                          f"  Tax paid: NIS {tax:,.0f}  "
                          f"  After: NIS {portfolio:,.0f}")
            year_start_portfolio = portfolio
            current_year = dt.year

        dd = (peak - portfolio) / peak
        peak = max(peak, portfolio)
        daily_snap.append(dict(
            date=dt, portfolio=round(portfolio, 2),
            drawdown=round(dd, 4),
            day_trades=day_trades,
        ))
        prev_date = dt

    # ── 6. Results ────────────────────────────────────────────────────────────
    trades_df = pd.DataFrame(trades)
    snaps_df  = pd.DataFrame(daily_snap)

    if trades_df.empty:
        print("No trades executed — check data and signal parameters.")
        return {}

    final_val   = snaps_df["portfolio"].iloc[-1]
    total_ret   = (final_val - INITIAL_CASH) / INITIAL_CASH
    n_years     = (pd.Timestamp(SIM_END) - pd.Timestamp(SIM_START)).days / 365.25
    cagr        = (final_val / INITIAL_CASH) ** (1 / n_years) - 1
    max_dd      = snaps_df["drawdown"].max()
    sharpe      = _sharpe(snaps_df["portfolio"])

    wins        = trades_df[trades_df["net_pnl"] > 0]
    losses      = trades_df[trades_df["net_pnl"] <= 0]
    win_rate    = len(wins) / len(trades_df)
    avg_win_pct = wins["gross_pct"].mean() if len(wins) > 0 else 0.0
    avg_loss_pct= losses["gross_pct"].mean() if len(losses) > 0 else 0.0
    outcome_cnt = trades_df["outcome"].value_counts().to_dict()

    results = dict(
        initial_cash    = INITIAL_CASH,
        final_portfolio = round(final_val, 2),
        total_return_pct= round(total_ret * 100, 2),
        cagr_pct        = round(cagr * 100, 2),
        sharpe          = round(sharpe, 2),
        max_drawdown_pct= round(max_dd * 100, 2),
        total_trades    = len(trades_df),
        win_rate_pct    = round(win_rate * 100, 1),
        avg_win_pct     = round(avg_win_pct, 3),
        avg_loss_pct    = round(avg_loss_pct, 3),
        outcome_counts  = outcome_cnt,
        trades_df       = trades_df,
        snaps_df        = snaps_df,
    )

    if verbose:
        _print_results(results)

    return results


def _sharpe(portfolio: pd.Series, rf: float = 0.038) -> float:
    ret = portfolio.pct_change().dropna()
    if len(ret) < 20:
        return 0.0
    excess = ret - rf / 252
    std = excess.std()
    return float(excess.mean() / std * np.sqrt(252)) if std > 0 else 0.0


def _print_results(r: dict):
    sep = "=" * 60
    print(f"\n{sep}")
    print("  TASE SCALPER -- 5-YEAR BACKTEST RESULTS")
    print(sep)
    print(f"  Period:          {SIM_START}  ->  {SIM_END}")
    print(f"  Universe:        {len(SCALP_UNIVERSE)} stocks (TASE most liquid)")
    print(f"  Strategy:        ORB Momentum Breakout (daily bar simulation)")
    print(f"  Initial capital: NIS {r['initial_cash']:,.0f}")
    print(f"  Final portfolio: NIS {r['final_portfolio']:,.0f}")
    print(f"  Total return:    {r['total_return_pct']:+.2f}%")
    print(f"  CAGR:            {r['cagr_pct']:+.2f}%")
    print(f"  Sharpe ratio:    {r['sharpe']:.2f}")
    print(f"  Max drawdown:    {r['max_drawdown_pct']:.2f}%")
    print(f"  Total trades:    {r['total_trades']}")
    print(f"  Win rate:        {r['win_rate_pct']:.1f}%")
    print(f"  Avg win:         {r['avg_win_pct']:+.3f}%")
    print(f"  Avg loss:        {r['avg_loss_pct']:+.3f}%")
    print(f"\n  Exit breakdown:")
    for outcome, cnt in sorted(r["outcome_counts"].items(), key=lambda x: -x[1]):
        pct_share = cnt / r["total_trades"] * 100
        print(f"    {outcome:<20} {cnt:>5}  ({pct_share:.1f}%)")
    print(sep)
    print(f"\n  Note: backtest uses daily OHLCV bars to simulate intraday ORB.")
    print(f"  Live trading via Interactive Brokers uses real 5-second bars.")


# ─── Chart ────────────────────────────────────────────────────────────────────
def plot_results(r: dict, save_path: str = "scalper_backtest.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        snaps = r["snaps_df"].copy()
        snaps["date"] = pd.to_datetime(snaps["date"])
        snaps = snaps.set_index("date")

        fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                                 gridspec_kw={"height_ratios": [3, 1, 1]})
        fig.suptitle(
            f"TASE ORB Scalper | 5-Year Backtest | "
            f"Return: {r['total_return_pct']:+.1f}% | "
            f"CAGR: {r['cagr_pct']:+.1f}% | "
            f"Sharpe: {r['sharpe']:.2f}",
            fontsize=12,
        )

        # Portfolio
        axes[0].plot(snaps.index, snaps["portfolio"],
                     color="royalblue", lw=1.5, label="Portfolio (NIS)")
        axes[0].axhline(INITIAL_CASH, color="gray", ls="--", lw=0.8,
                        label="Initial 100,000")
        axes[0].set_ylabel("Portfolio (NIS)")
        axes[0].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Drawdown
        axes[1].fill_between(snaps.index, snaps["drawdown"] * 100, 0,
                             color="crimson", alpha=0.5)
        axes[1].axhline(15, color="red", ls="--", lw=0.8,
                        label="15% guardrail")
        axes[1].set_ylabel("Drawdown %")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # Daily trades
        axes[2].bar(snaps.index, snaps["day_trades"],
                    color="steelblue", alpha=0.6, width=1.5)
        axes[2].set_ylabel("Trades / day")
        axes[2].grid(True, alpha=0.3)

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved -> {save_path}")
        return save_path

    except Exception as e:
        print(f"Chart error (non-fatal): {e}")
        return None


if __name__ == "__main__":
    results = run(verbose=True)
    if results:
        plot_results(results)
