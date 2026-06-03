"""
TASE Intraday VWAP Scalping Strategy
======================================
True day-trading: open and close ALL positions within the same trading day.
No overnight exposure. Trades last 5-30 minutes.

Strategy logic:
  VWAP = cumulative (price × volume) / cumulative volume, reset each morning.

  LONG  entry:  price drops ≥ ENTRY_BAND below VWAP (oversold dip)
                AND last 2 bars are recovering (close rising)
                AND volume on entry bar > 1.5x 20-bar avg (conviction)
  SHORT entry:  mirror image above VWAP
                (TASE allows short selling on blue-chip stocks)

  Exit:
    1. Target: price returns to VWAP midpoint
    2. Stop:   -STOP_PCT from entry (hard stop)
    3. Time:   max TIME_LIMIT bars (30 min on 5-min bars)
    4. EOD:    close everything 45 min before market close

Broker: Interactive Brokers (TWS API)
  - Commission: ~0.046% per side (min $2.50 ≈ 9 NIS)
  - Much cheaper than Israeli retail brokers (0.08-0.15%/side)

Backtest: uses actual 5-minute Yahoo Finance data (up to 60 days available).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

# Most liquid TASE stocks — best for intraday (tight spreads, high volume)
INTRADAY_UNIVERSE = [
    "POLI.TA",   # Bank Hapoalim — #1 by volume
    "LUMI.TA",   # Bank Leumi
    "MZTF.TA",   # Mizrahi-Tefahot
    "DSCT.TA",   # Bank Discount
    "TEVA.TA",   # Teva Pharmaceutical
    "ICL.TA",    # ICL Group
    "ESLT.TA",   # Elbit Systems (defense)
    "NICE.TA",   # NICE Systems
]

INITIAL_CAPITAL = 100_000.0   # NIS
MAX_POSITIONS   = 3            # simultaneous open intraday positions
POSITION_SIZE   = 0.25         # 25% of portfolio per trade

# Strategy parameters
ENTRY_BAND   = 0.006   # 0.6% deviation from VWAP to enter (only clear outliers)
TARGET_ABOVE = 0.001   # target 0.1% PAST vwap (slight overshoot for cushion)
STOP_PCT     = 0.004   # 0.4% stop loss from entry
TIME_LIMIT   = 8       # max 8 bars (= 40 minutes on 5-min bars)
VOLUME_MULT  = 1.5     # entry bar volume must be > 1.5× recent avg (conviction)
LONG_ONLY    = True    # True = only LONG trades (TASE short-selling is complex)

# Market hours (TASE, UTC)
MARKET_OPEN_UTC  = "07:55"   # 09:59 IL (UTC+2 winter, UTC+3 summer, ~07:55-08:55 UTC)
EOD_CUTOFF_BARS  = 9         # close all positions 9 bars (45 min) before market close
TASE_CLOSE_BAR   = "14:15"   # approx 5-min bar at which market closes (UTC)

# Costs (Interactive Brokers TASE)
COMMISSION_PCT = 0.00046    # 0.046% per side (IB minimum $2.50 per order)
COMMISSION_MIN = 9.0        # NIS (≈ $2.50) minimum per order
SLIPPAGE_PCT   = 0.0003     # 0.03% slippage for liquid stocks (1 tick typically)
TAX_RATE       = 0.25       # 25% Israeli capital gains (applied annually)

# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Trade:
    ticker:      str
    direction:   Literal["long", "short"]
    entry_time:  pd.Timestamp
    entry_price: float
    exit_time:   pd.Timestamp
    exit_price:  float
    qty:         float           # shares
    position_val: float          # NIS at entry
    gross_pnl:   float
    commission:  float
    net_pnl:     float
    outcome:     str             # "target" | "stop" | "time" | "eod"
    vwap_at_entry: float
    deviation_pct: float         # how far from VWAP at entry

@dataclass
class OpenPosition:
    ticker:      str
    direction:   Literal["long", "short"]
    entry_time:  pd.Timestamp
    entry_bar:   int
    entry_price: float
    qty:         float
    position_val: float
    stop_price:  float
    target_price: float
    vwap_at_entry: float
    deviation_pct: float

# ─── VWAP calculation ─────────────────────────────────────────────────────────

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Rolling VWAP, reset at the start of each trading day.
    Uses typical price = (H + L + C) / 3.
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_vol   = typical * df["Volume"]

    vwap = pd.Series(index=df.index, dtype=float)
    for date, group in df.groupby(df.index.date):
        idx   = group.index
        cum_v = df.loc[idx, "Volume"].cumsum()
        cum_tp= tp_vol.loc[idx].cumsum()
        vwap.loc[idx] = cum_tp / cum_v.replace(0, np.nan)

    return vwap


def compute_vol_ma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling average volume over `window` bars."""
    return df["Volume"].rolling(window, min_periods=5).mean()


# ─── Signal detection ─────────────────────────────────────────────────────────

def detect_signal(
    bar: pd.Series,
    prev_bar: pd.Series,
    prev2_bar: pd.Series,
    vwap: float,
    vol_ma: float,
) -> Literal["long", "short", None]:
    """
    Detect a VWAP mean-reversion entry signal.

    LONG signal (buy dip):
      - Price is ENTRY_BAND% below VWAP
      - Last 2 bars recovering (close rising)
      - Volume confirmation

    SHORT signal (sell spike):
      - Price is ENTRY_BAND% above VWAP
      - Last 2 bars declining
      - Volume confirmation
    """
    if pd.isna(vwap) or vwap <= 0:
        return None

    price   = bar["Close"]
    dev     = (price - vwap) / vwap

    # Volume must be above average
    vol_ok = (vol_ma > 0) and (bar["Volume"] > vol_ma * VOLUME_MULT)

    # Price recovery confirmation (last 2 bars moving in signal direction)
    c0 = bar["Close"]
    c1 = prev_bar["Close"]   if prev_bar is not None else c0
    c2 = prev2_bar["Close"]  if prev2_bar is not None else c1

    if dev <= -ENTRY_BAND and vol_ok:
        # LONG: price dipped below VWAP band AND 2+ bars recovering
        if c0 > c1 and c1 >= c2:   # 2-bar recovery confirmation
            return "long"

    if not LONG_ONLY and dev >= ENTRY_BAND and vol_ok:
        # SHORT: price spiked above VWAP band AND 2+ bars rolling over
        if c0 < c1 and c1 <= c2:
            return "short"

    return None


# ─── Backtest engine ──────────────────────────────────────────────────────────

class VWAPBacktest:
    def __init__(self, capital: float = INITIAL_CAPITAL):
        self.capital    = capital
        self.portfolio  = capital
        self.peak       = capital
        self.positions: dict[str, OpenPosition] = {}
        self.trades:    list[Trade]             = []
        self._annual_pnl: dict[int, float]      = {}

    def run(self, all_bars: dict[str, pd.DataFrame], verbose: bool = True) -> dict:
        """
        all_bars: {ticker: 5-min OHLCV DataFrame with DatetimeIndex}
        """
        # Compute VWAP and volume MA for each ticker
        vwaps   = {}
        vol_mas = {}
        for ticker, df in all_bars.items():
            vwaps[ticker]   = compute_vwap(df)
            vol_mas[ticker] = compute_vol_ma(df)

        # Build unified timeline
        all_times = sorted({t for df in all_bars.values() for t in df.index})
        if verbose:
            print(f"  Simulation: {all_times[0]} -> {all_times[-1]}")
            print(f"  Total bars: {len(all_times)}")

        bar_idx = 0
        for ts in all_times:
            bar_idx += 1

            # ── Update open positions ──────────────────────────────────────────
            for ticker in list(self.positions.keys()):
                if ticker not in all_bars or ts not in all_bars[ticker].index:
                    continue
                pos = self.positions[ticker]
                bar = all_bars[ticker].loc[ts]
                vwap = vwaps[ticker].get(ts, np.nan)

                # Check exits (in priority order)
                exit_price, outcome = self._check_exit(pos, bar, ts, bar_idx, vwap)
                if exit_price is not None:
                    self._close_position(pos, exit_price, ts, outcome)
                    del self.positions[ticker]

            # ── Check for new signals ─────────────────────────────────────────
            if len(self.positions) < MAX_POSITIONS:
                for ticker, df in all_bars.items():
                    if ticker in self.positions or ts not in df.index:
                        continue

                    bar  = df.loc[ts]
                    idx  = df.index.get_loc(ts)
                    prev = df.iloc[idx-1] if idx >= 1 else None
                    prev2= df.iloc[idx-2] if idx >= 2 else None
                    vwap = vwaps[ticker].get(ts, np.nan)
                    volma= vol_mas[ticker].get(ts, np.nan)

                    # Skip first hour of each day (VWAP not yet stable)
                    day_bars = df[df.index.date == ts.date()]
                    bar_in_day = day_bars.index.get_loc(ts) if ts in day_bars.index else 0
                    if bar_in_day < 12:   # skip first 60 minutes (12 bars × 5 min)
                        continue

                    # Skip EOD (last 9 bars of day ≈ 45 min before close)
                    if bar_in_day >= len(day_bars) - EOD_CUTOFF_BARS:
                        continue

                    sig = detect_signal(bar, prev, prev2, vwap, volma)
                    if sig:
                        self._open_position(ticker, sig, bar, ts, bar_idx, vwap)

            # ── EOD forced close ──────────────────────────────────────────────
            # Close all if this is the last bar of the day
            for ticker in list(self.positions.keys()):
                if ticker not in all_bars or ts not in all_bars[ticker].index:
                    continue
                df = all_bars[ticker]
                day_bars = df[df.index.date == ts.date()]
                if len(day_bars) > 0 and ts == day_bars.index[-1]:
                    bar = df.loc[ts]
                    self._close_position(self.positions[ticker],
                                         bar["Close"], ts, "eod")
                    del self.positions[ticker]

        # Apply annual tax on cumulative gains
        self._apply_annual_tax(verbose=verbose)
        return self._compile_results()

    def _open_position(self, ticker: str, direction: str,
                       bar: pd.Series, ts: pd.Timestamp,
                       bar_idx: int, vwap: float):
        price      = bar["Close"]
        pos_val    = self.portfolio * POSITION_SIZE
        # IB minimum commission check
        comm_est   = max(pos_val * COMMISSION_PCT, COMMISSION_MIN)
        if pos_val > self.portfolio * 0.95:
            return  # insufficient capital

        qty = pos_val / price
        dev = (price - vwap) / vwap

        if direction == "long":
            stop_price   = price * (1 - STOP_PCT)
            target_price = vwap * (1 + TARGET_ABOVE)  # target PAST vwap (overshoot)
        else:
            stop_price   = price * (1 + STOP_PCT)
            target_price = vwap * (1 - TARGET_ABOVE)  # target PAST vwap downside

        self.positions[ticker] = OpenPosition(
            ticker=ticker, direction=direction,
            entry_time=ts, entry_bar=bar_idx,
            entry_price=price, qty=qty, position_val=pos_val,
            stop_price=stop_price, target_price=target_price,
            vwap_at_entry=vwap, deviation_pct=dev * 100,
        )

    def _check_exit(self, pos: OpenPosition, bar: pd.Series,
                    ts: pd.Timestamp, bar_idx: int,
                    vwap: float) -> tuple[float | None, str | None]:
        """Return (exit_price, outcome) or (None, None) to hold."""
        high  = bar["High"]
        low   = bar["Low"]
        close = bar["Close"]
        bars_held = bar_idx - pos.entry_bar

        if pos.direction == "long":
            # Stop hit
            if low <= pos.stop_price:
                return pos.stop_price, "stop"
            # Target hit (price returned toward VWAP)
            if high >= pos.target_price and bars_held >= 1:
                return pos.target_price, "target"
        else:  # short
            if high >= pos.stop_price:
                return pos.stop_price, "stop"
            if low <= pos.target_price and bars_held >= 1:
                return pos.target_price, "target"

        # Time limit
        if bars_held >= TIME_LIMIT:
            return close, "time"

        return None, None

    def _close_position(self, pos: OpenPosition, exit_price: float,
                        ts: pd.Timestamp, outcome: str):
        commission = max(
            (pos.position_val + pos.position_val * abs(exit_price / pos.entry_price - 1)) * COMMISSION_PCT,
            COMMISSION_MIN * 2   # buy + sell
        )
        slippage = pos.position_val * SLIPPAGE_PCT

        if pos.direction == "long":
            gross = pos.qty * (exit_price - pos.entry_price)
        else:
            gross = pos.qty * (pos.entry_price - exit_price)

        net = gross - commission - slippage
        self.portfolio += net

        # Track for annual tax
        yr = ts.year
        self._annual_pnl[yr] = self._annual_pnl.get(yr, 0) + net

        self.peak = max(self.peak, self.portfolio)
        self.trades.append(Trade(
            ticker=pos.ticker, direction=pos.direction,
            entry_time=pos.entry_time, entry_price=pos.entry_price,
            exit_time=ts, exit_price=exit_price,
            qty=round(pos.qty, 2), position_val=round(pos.position_val, 2),
            gross_pnl=round(gross, 2), commission=round(commission, 2),
            net_pnl=round(net, 2), outcome=outcome,
            vwap_at_entry=round(pos.vwap_at_entry, 4),
            deviation_pct=round(pos.deviation_pct, 3),
        ))

    def _apply_annual_tax(self, verbose: bool = True):
        """Deduct 25% on net annual profits."""
        cumulative_tax = 0
        for yr in sorted(self._annual_pnl):
            profit = self._annual_pnl[yr]
            if profit > 0:
                tax = profit * TAX_RATE
                self.portfolio -= tax
                cumulative_tax += tax
                if verbose:
                    print(f"  [{yr} tax] profit: {profit:,.0f} NIS  "
                          f"tax: {tax:,.0f} NIS  after: {self.portfolio:,.0f} NIS")

    def _compile_results(self) -> dict:
        if not self.trades:
            return {}

        df = pd.DataFrame(self.trades)
        n  = len(df)
        wins   = df[df["net_pnl"] > 0]
        losses = df[df["net_pnl"] <= 0]

        win_rate    = len(wins) / n * 100
        avg_win     = wins["net_pnl"].mean()  if len(wins)   else 0
        avg_loss    = losses["net_pnl"].mean() if len(losses) else 0
        total_comm  = df["commission"].sum()
        total_gross = df["gross_pnl"].sum()
        total_net   = df["net_pnl"].sum()

        # Days in backtest
        days    = (df["exit_time"].max() - df["entry_time"].min()).days or 1
        n_years = days / 365.25
        daily_trades = n / (days * 5/7)  # approx trading days

        final   = self.portfolio
        ret_pct = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        cagr    = ((final / INITIAL_CAPITAL) ** (1/n_years) - 1) * 100 if n_years > 0 else 0

        # Sharpe (simplified)
        daily_net = df.groupby(df["exit_time"].dt.date)["net_pnl"].sum()
        daily_ret = daily_net / INITIAL_CAPITAL
        sharpe    = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                     if daily_ret.std() > 0 else 0)

        outcome_ct = df["outcome"].value_counts().to_dict()
        direction_ct = df["direction"].value_counts().to_dict()

        # Annualized projections (from actual data)
        annual_trades = n / n_years
        annual_net    = total_net / n_years

        return dict(
            initial_capital = INITIAL_CAPITAL,
            final_portfolio = round(final, 2),
            total_return_pct= round(ret_pct, 2),
            cagr_pct        = round(cagr, 2),
            sharpe          = round(sharpe, 2),
            backtest_days   = days,
            total_trades    = n,
            daily_avg_trades= round(daily_trades, 1),
            win_rate_pct    = round(win_rate, 1),
            avg_win_nis     = round(avg_win, 1),
            avg_loss_nis    = round(avg_loss, 1),
            rr_ratio        = round(abs(avg_win/avg_loss), 2) if avg_loss else 0,
            total_gross_nis = round(total_gross, 1),
            total_commission= round(total_comm, 1),
            total_net_nis   = round(total_net, 1),
            annual_trades_proj  = round(annual_trades, 0),
            annual_profit_proj  = round(annual_net, 0),
            annual_return_proj  = round(annual_net / INITIAL_CAPITAL * 100, 1),
            outcome_counts  = outcome_ct,
            direction_counts= direction_ct,
            trades_df       = df,
        )


# ─── Data download ────────────────────────────────────────────────────────────

def download_intraday(tickers: list[str],
                      period: str = "60d",
                      interval: str = "5m",
                      verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Download 5-minute bars from Yahoo Finance (max 60 days for TASE)."""
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval=interval,
                             progress=False, multi_level_index=False,
                             auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.capitalize() if c.lower() in
                          ("open","high","low","close","volume") else c
                          for c in df.columns]
            # Normalize timezone (TASE data comes in UTC)
            if df.index.tzinfo is not None:
                df.index = df.index.tz_convert("Asia/Jerusalem")
            df = df.dropna(subset=["Close", "Volume"])
            if len(df) >= 50:
                data[t] = df
                if verbose:
                    print(f"  {t}: {len(df)} bars  "
                          f"({df.index[0].date()} -> {df.index[-1].date()})")
        except Exception as e:
            if verbose:
                print(f"  {t}: failed ({e})")
    return data


# ─── Print results ────────────────────────────────────────────────────────────

def print_results(r: dict, note: str = ""):
    if not r:
        print("No results.")
        return
    sep = "=" * 65
    print(f"\n{sep}")
    print("  TASE INTRADAY VWAP SCALPER — BACKTEST RESULTS")
    if note:
        print(f"  {note}")
    print(sep)
    print(f"  Data window:        {r['backtest_days']} calendar days "
          f"(~{int(r['backtest_days']*5/7)} trading days)")
    print(f"  Initial capital:    NIS {r['initial_capital']:,.0f}")
    print(f"  Final portfolio:    NIS {r['final_portfolio']:,.0f}")
    print(f"  Return (period):    {r['total_return_pct']:+.2f}%")
    print(f"  Annualized return:  {r['cagr_pct']:+.2f}%")
    print(f"  Sharpe ratio:       {r['sharpe']:.2f}")
    print(f"")
    print(f"  Total trades:       {r['total_trades']}")
    print(f"  Trades / day (avg): {r['daily_avg_trades']}")
    print(f"  Win rate:           {r['win_rate_pct']:.1f}%")
    print(f"  Avg win:            NIS {r['avg_win_nis']:+.1f} per trade")
    print(f"  Avg loss:           NIS {r['avg_loss_nis']:+.1f} per trade")
    print(f"  Risk:Reward ratio:  {r['rr_ratio']:.2f}:1")
    print(f"")
    print(f"  Total gross PnL:    NIS {r['total_gross_nis']:+,.0f}")
    print(f"  Total commission:   NIS {r['total_commission']:,.0f}")
    print(f"  Total net PnL:      NIS {r['total_net_nis']:+,.0f}")
    print(f"")
    print(f"  --- Projected annual (extrapolated from {r['backtest_days']}d data) ---")
    print(f"  Trades/year:        ~{int(r['annual_trades_proj']):,}")
    print(f"  Annual profit:      NIS {r['annual_profit_proj']:+,.0f}")
    print(f"  Annual return:      {r['annual_return_proj']:+.1f}%")
    print(f"")
    print(f"  Exit breakdown:")
    for k, v in sorted(r["outcome_counts"].items(), key=lambda x: -x[1]):
        pct = v / r["total_trades"] * 100
        print(f"    {k:<15} {v:>5}  ({pct:.1f}%)")
    print(f"  Direction: {r['direction_counts']}")
    print(sep)
    print(f"  Note: backtest uses IB commission model (0.046%/side, min $2.50).")
    print(f"  Israeli retail brokers (0.08-0.15%/side) would cut returns ~40-60%.")


# ─── Chart ────────────────────────────────────────────────────────────────────

def plot_results(r: dict, save_path: str = "intraday_backtest.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        df = r["trades_df"].copy()
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df = df.sort_values("exit_time")

        # Equity curve
        equity = INITIAL_CAPITAL + df["net_pnl"].cumsum()
        equity.index = df["exit_time"].values

        fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                                 gridspec_kw={"height_ratios": [3, 1.2, 1]})

        ann_ret = r['annual_return_proj']
        fig.suptitle(
            f"TASE VWAP Intraday Scalper  |  "
            f"Win rate: {r['win_rate_pct']:.1f}%  |  "
            f"Projected annual: {ann_ret:+.1f}%  |  "
            f"Sharpe: {r['sharpe']:.2f}",
            fontsize=11
        )

        # Equity
        axes[0].plot(df["exit_time"].values, equity.values,
                     color="royalblue", lw=1.5)
        axes[0].axhline(INITIAL_CAPITAL, color="gray", ls="--", lw=0.8)
        axes[0].set_ylabel("Portfolio (NIS)")
        axes[0].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        axes[0].grid(True, alpha=0.3)

        # Per-trade PnL
        colors = ["limegreen" if p > 0 else "tomato" for p in df["net_pnl"]]
        axes[1].bar(range(len(df)), df["net_pnl"].values,
                    color=colors, alpha=0.7, width=0.8)
        axes[1].axhline(0, color="black", lw=0.5)
        axes[1].set_ylabel("Net PnL per trade (NIS)")
        axes[1].grid(True, alpha=0.3)

        # Deviation at entry
        axes[2].scatter(range(len(df)), df["deviation_pct"].values,
                        c=colors, alpha=0.6, s=20)
        axes[2].axhline(0, color="black", lw=0.5)
        axes[2].axhline(ENTRY_BAND * 100, color="green", ls="--", lw=0.7,
                        label=f"+{ENTRY_BAND*100:.1f}% entry band")
        axes[2].axhline(-ENTRY_BAND * 100, color="red", ls="--", lw=0.7,
                        label=f"-{ENTRY_BAND*100:.1f}% entry band")
        axes[2].set_ylabel("VWAP deviation at entry %")
        axes[2].legend(fontsize=7)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved -> {save_path}")
        return save_path
    except Exception as e:
        print(f"Chart error: {e}")
        return None
