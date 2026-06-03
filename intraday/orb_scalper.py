"""
TASE Intraday ORB (Opening Range Breakout) Scalper
====================================================
True day-trading: buy/sell breakouts from the first 30-minute range.
All positions closed within the same day. No overnight exposure.

Strategy:
  1. Opening range (OR): 09:59-10:29 IL — track high/low of first 30 min.
  2. After 10:29: if price breaks ABOVE range_high with volume -> BUY
                  if price breaks BELOW range_low with volume  -> SELL (long-only mode: skip)
  3. Exit:
     a. Target  = entry + range_height × TARGET_MULT (e.g. 1.5× range above breakout)
     b. Stop    = range_low (for longs) — below the range = breakout failed
     c. VWAP    = exit if price falls back below VWAP (adds mean-reversion filter)
     d. EOD     = close everything by 15:45 IL (90 min before close)

Broker: Interactive Brokers (TWS API)
  Commission: ~0.046% per side (min $2.50 ≈ 9 NIS)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Universe ──────────────────────────────────────────────────────────────────
INTRADAY_UNIVERSE = [
    "POLI.TA",   # Bank Hapoalim
    "LUMI.TA",   # Bank Leumi
    "MZTF.TA",   # Mizrahi-Tefahot
    "DSCT.TA",   # Bank Discount
    "TEVA.TA",   # Teva Pharmaceutical
    "ICL.TA",    # ICL Group
    "ESLT.TA",   # Elbit Systems
    "NICE.TA",   # NICE Systems
    "CAMT.TA",   # Camtek
    "NVMI.TA",   # Nova Measuring
]

INITIAL_CAPITAL = 100_000.0

# ─── Strategy parameters ──────────────────────────────────────────────────────
OR_BARS       = 6          # opening range = first 6 bars of 5 min = 30 minutes
BREAKOUT_PCT  = 0.001      # require price > OR_high × 1.001 (filter false breaks)
TARGET_MULT   = 1.5        # target = entry + OR_height × 1.5
MAX_POSITIONS = 3          # max simultaneous positions
POSITION_SIZE = 0.25       # 25% of portfolio per trade
VOLUME_MULT   = 1.3        # breakout bar volume > 1.3× OR avg volume
LONG_ONLY     = True       # True = only buy breakouts (no shorts — TASE restriction)

# Market timing (IL = UTC+2/+3)
OR_END_BAR    = 6          # bar index (0-based) when opening range ends
EOD_CUTOFF_H  = 15         # stop new trades at 15:45 IL
EOD_CUTOFF_M  = 45
FORCE_CLOSE_H = 16         # force-close all by 16:00 IL
FORCE_CLOSE_M = 30

# Costs (Interactive Brokers)
COMMISSION_PCT = 0.00046
COMMISSION_MIN = 9.0        # NIS per order
SLIPPAGE_PCT   = 0.0003
TAX_RATE       = 0.25


# ─── Data structures ──────────────────────────────────────────────────────────
@dataclass
class ORState:
    """Tracks opening range per ticker per day."""
    or_high:   float = 0.0
    or_low:    float = float("inf")
    or_avg_vol: float = 0.0
    or_complete: bool = False

@dataclass
class ORTrade:
    ticker:       str
    direction:    str
    entry_time:   pd.Timestamp
    entry_price:  float
    exit_time:    pd.Timestamp
    exit_price:   float
    qty:          float
    position_val: float
    stop_price:   float
    target_price: float
    or_height:    float
    gross_pnl:    float
    commission:   float
    net_pnl:      float
    outcome:      str    # "target" | "stop" | "eod" | "vwap"


# ─── VWAP helper (daily reset) ────────────────────────────────────────────────
def compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_vol  = typical * df["Volume"]
    vwap    = pd.Series(index=df.index, dtype=float)
    for _date, grp in df.groupby(df.index.date):
        idx       = grp.index
        cum_v     = df.loc[idx, "Volume"].cumsum()
        cum_tp    = tp_vol.loc[idx].cumsum()
        vwap.loc[idx] = cum_tp / cum_v.replace(0, np.nan)
    return vwap


# ─── Backtest engine ──────────────────────────────────────────────────────────
class ORBBacktest:
    def __init__(self, capital: float = INITIAL_CAPITAL):
        self.capital    = capital
        self.portfolio  = capital
        self.peak       = capital
        self.positions: dict[str, ORTrade] = {}   # open positions (keyed by ticker)
        self.trades:    list[ORTrade]      = []
        self._annual_pnl: dict[int, float] = {}

    def run(self, all_bars: dict[str, pd.DataFrame],
            index_daily: pd.DataFrame | None = None,
            verbose: bool = True) -> dict:
        """
        index_daily: optional daily OHLCV for ^TA125.TA.
                     If provided, only enter longs on days when index is UP.
        """
        vwaps = {t: compute_vwap(df) for t, df in all_bars.items()}

        # Build index daily-change map {date: True if index up}
        index_up: dict = {}
        if index_daily is not None:
            idx = index_daily["Close"].copy()
            idx.index = pd.to_datetime(idx.index)
            for i in range(1, len(idx)):
                d = idx.index[i].date()
                index_up[d] = (idx.iloc[i] >= idx.iloc[i-1])

        all_dates = sorted({d for df in all_bars.values() for d in df.index.date})

        if verbose:
            print(f"  Days to process: {len(all_dates)}")

        skipped = 0
        for day in all_dates:
            # Regime filter: skip if index down today
            if index_up and not index_up.get(day, True):
                skipped += 1
                continue
            self._process_day(day, all_bars, vwaps, verbose=False)

        if verbose and skipped:
            print(f"  Skipped {skipped} down-index days (regime filter)")

        self._apply_annual_tax(verbose=verbose)
        return self._compile_results()

    def _process_day(self, day: pd.Timestamp, all_bars: dict,
                     vwaps: dict, verbose: bool):
        or_states: dict[str, ORState] = {}
        open_pos:  dict[str, dict]    = {}   # ticker -> {entry_price, stop, target, qty, posval, or_height, dir}

        # Collect bars for this day across all tickers
        target_date = day.date() if hasattr(day, "date") else day
        day_data: dict[str, pd.DataFrame] = {}
        for ticker, df in all_bars.items():
            mask   = [d == target_date for d in df.index.date]
            day_df = df[mask]
            if len(day_df) >= OR_END_BAR + 2:
                day_data[ticker] = day_df
                or_states[ticker] = ORState()

        if not day_data:
            return

        # Get unified timeline for this day
        day_times = sorted({t for df in day_data.values() for t in df.index})

        for i, ts in enumerate(day_times):
            hour_il = ts.hour    # index is already in Asia/Jerusalem TZ
            min_il  = ts.minute

            # Determine bar index within day per ticker
            for ticker, df in day_data.items():
                if ts not in df.index:
                    continue

                bar       = df.loc[ts]
                bar_in_day= list(df.index).index(ts)
                st        = or_states[ticker]
                vwap_now  = vwaps[ticker].get(ts, np.nan)

                # ── Build Opening Range (bars 0-5, i.e. 09:59-10:29) ──────────
                if bar_in_day < OR_END_BAR:
                    if bar["High"] > st.or_high:
                        st.or_high = bar["High"]
                    if bar["Low"] < st.or_low:
                        st.or_low = bar["Low"]
                    st.or_avg_vol = (
                        (st.or_avg_vol * bar_in_day + bar["Volume"]) / (bar_in_day + 1)
                    )
                    continue
                else:
                    st.or_complete = True

                if not st.or_complete:
                    continue

                # ── Manage open position ───────────────────────────────────────
                if ticker in open_pos:
                    pos = open_pos[ticker]
                    high, low, close = bar["High"], bar["Low"], bar["Close"]

                    eod_hit = (hour_il > FORCE_CLOSE_H or
                               (hour_il == FORCE_CLOSE_H and min_il >= FORCE_CLOSE_M))

                    exit_p, outcome = None, None

                    if pos["dir"] == "long":
                        if low <= pos["stop"]:
                            exit_p, outcome = pos["stop"], "stop"
                        elif high >= pos["target"]:
                            exit_p, outcome = pos["target"], "target"

                    if eod_hit and outcome is None:
                        exit_p, outcome = bar["Close"], "eod"

                    if exit_p is not None:
                        self._record_trade(pos, exit_p, ts, outcome)
                        del open_pos[ticker]

                # ── Check for new breakout signal ──────────────────────────────
                if ticker in open_pos:
                    continue
                if len(open_pos) >= MAX_POSITIONS:
                    continue

                # No new trades in last 90 min
                eod_cutoff = (hour_il > EOD_CUTOFF_H or
                              (hour_il == EOD_CUTOFF_H and min_il >= EOD_CUTOFF_M))
                if eod_cutoff:
                    continue

                or_h = st.or_high
                or_l = st.or_low
                or_height = or_h - or_l
                if or_height <= 0:
                    continue

                # Skip chaotic openings (OR height > 2% = too volatile/unclear)
                or_height_pct = or_height / or_h
                if or_height_pct > 0.02:
                    continue

                close = bar["Close"]
                vol   = bar["Volume"]

                # LONG: breakout above OR high
                if close > or_h * (1 + BREAKOUT_PCT):
                    if vol > st.or_avg_vol * VOLUME_MULT:
                        entry_p  = close
                        stop_p   = or_h - or_height * 0.5     # stop = mid-OR (tighter)
                        target_p = entry_p + or_height * TARGET_MULT
                        pos_val  = self.portfolio * POSITION_SIZE
                        qty      = pos_val / entry_p
                        open_pos[ticker] = {
                            "dir": "long", "entry_price": entry_p,
                            "stop": stop_p, "target": target_p,
                            "qty": qty, "pos_val": pos_val,
                            "or_height": or_height, "entry_time": ts,
                            "ticker": ticker,
                        }

                # SHORT: breakout below OR low (only if not LONG_ONLY)
                elif not LONG_ONLY and close < or_l * (1 - BREAKOUT_PCT):
                    if vol > st.or_avg_vol * VOLUME_MULT:
                        entry_p  = close
                        stop_p   = or_h
                        target_p = entry_p - or_height * TARGET_MULT
                        pos_val  = self.portfolio * POSITION_SIZE
                        qty      = pos_val / entry_p
                        open_pos[ticker] = {
                            "dir": "short", "entry_price": entry_p,
                            "stop": stop_p, "target": target_p,
                            "qty": qty, "pos_val": pos_val,
                            "or_height": or_height, "entry_time": ts,
                            "ticker": ticker,
                        }

        # ── EOD forced close for any remaining positions ───────────────────────
        for ticker, pos in list(open_pos.items()):
            df = day_data[ticker]
            last_bar = df.iloc[-1]
            self._record_trade(pos, last_bar["Close"], df.index[-1], "eod")

    def _record_trade(self, pos: dict, exit_price: float,
                      exit_time: pd.Timestamp, outcome: str):
        entry = pos["entry_price"]
        qty   = pos["qty"]
        posv  = pos["pos_val"]

        comm = max(posv * COMMISSION_PCT * 2, COMMISSION_MIN * 2)
        slip = posv * SLIPPAGE_PCT

        if pos["dir"] == "long":
            gross = qty * (exit_price - entry)
        else:
            gross = qty * (entry - exit_price)

        net = gross - comm - slip
        self.portfolio += net
        self.peak = max(self.peak, self.portfolio)

        yr = exit_time.year
        self._annual_pnl[yr] = self._annual_pnl.get(yr, 0) + net

        self.trades.append(ORTrade(
            ticker=pos["ticker"], direction=pos["dir"],
            entry_time=pos["entry_time"], entry_price=entry,
            exit_time=exit_time, exit_price=exit_price,
            qty=round(qty, 2), position_val=round(posv, 2),
            stop_price=round(pos["stop"], 4),
            target_price=round(pos["target"], 4),
            or_height=round(pos["or_height"], 4),
            gross_pnl=round(gross, 2), commission=round(comm, 2),
            net_pnl=round(net, 2), outcome=outcome,
        ))

    def _apply_annual_tax(self, verbose: bool = True):
        total_tax = 0
        for yr in sorted(self._annual_pnl):
            profit = self._annual_pnl[yr]
            if profit > 0:
                tax = profit * TAX_RATE
                self.portfolio -= tax
                total_tax += tax
                if verbose:
                    print(f"  [{yr} tax] profit: {profit:,.0f} NIS  "
                          f"tax: {tax:,.0f} NIS  after: {self.portfolio:,.0f} NIS")

    def _compile_results(self) -> dict:
        if not self.trades:
            return {}
        df = pd.DataFrame([t.__dict__ for t in self.trades])
        n  = len(df)
        wins   = df[df["net_pnl"] > 0]
        losses = df[df["net_pnl"] <= 0]

        win_rate = len(wins) / n * 100
        avg_win  = wins["net_pnl"].mean()  if len(wins)   else 0
        avg_loss = losses["net_pnl"].mean() if len(losses) else 0

        days    = (df["exit_time"].max() - df["entry_time"].min()).days or 1
        n_years = days / 365.25
        final   = self.portfolio
        ret_pct = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        cagr    = ((final / INITIAL_CAPITAL) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

        daily_net = df.groupby(df["exit_time"].dt.date)["net_pnl"].sum()
        daily_ret = daily_net / INITIAL_CAPITAL
        sharpe    = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                     if daily_ret.std() > 0 else 0)

        max_dd = 0.0
        peak   = INITIAL_CAPITAL
        eq     = INITIAL_CAPITAL
        for pnl in df["net_pnl"]:
            eq  += pnl
            peak = max(peak, eq)
            dd   = (peak - eq) / peak * 100
            max_dd = max(max_dd, dd)

        return dict(
            initial_capital  = INITIAL_CAPITAL,
            final_portfolio  = round(final, 2),
            total_return_pct = round(ret_pct, 2),
            cagr_pct         = round(cagr, 2),
            sharpe           = round(sharpe, 2),
            max_drawdown_pct = round(max_dd, 2),
            backtest_days    = days,
            total_trades     = n,
            daily_avg_trades = round(n / (days * 5/7), 2),
            win_rate_pct     = round(win_rate, 1),
            avg_win_nis      = round(avg_win, 1),
            avg_loss_nis     = round(avg_loss, 1),
            rr_ratio         = round(abs(avg_win / avg_loss), 2) if avg_loss else 0,
            total_gross_nis  = round(df["gross_pnl"].sum(), 1),
            total_commission = round(df["commission"].sum(), 1),
            total_net_nis    = round(df["net_pnl"].sum(), 1),
            annual_trades_proj  = round(n / n_years, 0),
            annual_profit_proj  = round(df["net_pnl"].sum() / n_years, 0),
            annual_return_proj  = round(df["net_pnl"].sum() / n_years / INITIAL_CAPITAL * 100, 1),
            outcome_counts   = df["outcome"].value_counts().to_dict(),
            direction_counts = df["direction"].value_counts().to_dict(),
            trades_df        = df,
        )


# ─── Data download ────────────────────────────────────────────────────────────
def download_intraday(tickers: list[str],
                      period: str = "60d",
                      interval: str = "5m",
                      verbose: bool = True) -> dict[str, pd.DataFrame]:
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
            if df.index.tzinfo is not None:
                df.index = df.index.tz_convert("Asia/Jerusalem")
            df = df.dropna(subset=["Close", "Volume"])
            if len(df) >= 30:
                data[t] = df
                if verbose:
                    print(f"  {t}: {len(df)} bars  "
                          f"({df.index[0].date()} -> {df.index[-1].date()})")
        except Exception as e:
            if verbose:
                print(f"  {t}: failed ({e})")
    return data


# ─── Output ───────────────────────────────────────────────────────────────────
def print_results(r: dict, note: str = ""):
    if not r:
        print("No results.")
        return
    sep = "=" * 65
    print(f"\n{sep}")
    print("  TASE ORB INTRADAY SCALPER — BACKTEST RESULTS")
    if note:
        print(f"  {note}")
    print(sep)
    print(f"  Data window:        {r['backtest_days']} calendar days "
          f"(~{int(r['backtest_days']*5/7)} trading days)")
    print(f"  Initial capital:    NIS {r['initial_capital']:,.0f}")
    print(f"  Final portfolio:    NIS {r['final_portfolio']:,.0f}")
    print(f"  Return (period):    {r['total_return_pct']:+.2f}%")
    print(f"  Annualized CAGR:    {r['cagr_pct']:+.2f}%")
    print(f"  Sharpe ratio:       {r['sharpe']:.2f}")
    print(f"  Max drawdown:       {r['max_drawdown_pct']:.2f}%")
    print()
    print(f"  Total trades:       {r['total_trades']}")
    print(f"  Trades / day (avg): {r['daily_avg_trades']}")
    print(f"  Win rate:           {r['win_rate_pct']:.1f}%")
    print(f"  Avg win:            NIS {r['avg_win_nis']:+.1f}")
    print(f"  Avg loss:           NIS {r['avg_loss_nis']:+.1f}")
    print(f"  Risk:Reward ratio:  {r['rr_ratio']:.2f}:1")
    print()
    print(f"  Gross PnL:          NIS {r['total_gross_nis']:+,.0f}")
    print(f"  Total commission:   NIS {r['total_commission']:,.0f}")
    print(f"  Total net PnL:      NIS {r['total_net_nis']:+,.0f}")
    print()
    print(f"  --- Projected annual (from {r['backtest_days']}d window) ---")
    print(f"  Trades/year:        ~{int(r['annual_trades_proj']):,}")
    print(f"  Annual profit:      NIS {r['annual_profit_proj']:+,.0f}")
    print(f"  Annual return:      {r['annual_return_proj']:+.1f}%")
    print()
    print(f"  Exit breakdown:")
    for k, v in sorted(r["outcome_counts"].items(), key=lambda x: -x[1]):
        pct = v / r["total_trades"] * 100
        print(f"    {k:<15} {v:>5}  ({pct:.1f}%)")
    print(f"  Direction: {r['direction_counts']}")
    print(sep)


def plot_results(r: dict, save_path: str = "orb_backtest.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = r["trades_df"].copy()
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df = df.sort_values("exit_time")
        equity = INITIAL_CAPITAL + df["net_pnl"].cumsum()

        fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                                 gridspec_kw={"height_ratios": [3, 1.2, 1]})

        fig.suptitle(
            f"TASE ORB Intraday Scalper  |  "
            f"Win rate: {r['win_rate_pct']:.1f}%  |  "
            f"Annual proj: {r['annual_return_proj']:+.1f}%  |  "
            f"Sharpe: {r['sharpe']:.2f}",
            fontsize=11
        )

        axes[0].plot(df["exit_time"].values, equity.values,
                     color="royalblue", lw=1.5)
        axes[0].axhline(INITIAL_CAPITAL, color="gray", ls="--", lw=0.8,
                        label="Initial capital")
        axes[0].set_ylabel("Portfolio (NIS)")
        axes[0].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        colors = ["limegreen" if p > 0 else "tomato" for p in df["net_pnl"]]
        axes[1].bar(range(len(df)), df["net_pnl"].values,
                    color=colors, alpha=0.7, width=0.8)
        axes[1].axhline(0, color="black", lw=0.5)
        axes[1].set_ylabel("Net PnL / trade (NIS)")
        axes[1].grid(True, alpha=0.3)

        # OR height distribution
        axes[2].bar(range(len(df)), df["or_height"].values,
                    color="steelblue", alpha=0.6, width=0.8)
        axes[2].set_ylabel("Opening range height")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved -> {save_path}")
    except Exception as e:
        print(f"Chart error: {e}")
