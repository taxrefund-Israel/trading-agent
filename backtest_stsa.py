"""
TASE Short-Term Swing Alpha (STSA) — runs 4 strategy variants
and compares them vs TA-125 buy-and-hold.

STSA focuses on:
  - Concentrated book: 5 positions x 22% (vs v9's 10 x 12%)
  - Short holds: 10-day min (vs v9's 40-day) — catch & exit faster
  - Tighter ATR trail: 2.5x (vs v9's 3.5x) — protect gains early
  - RS threshold: 5% outperformance over index (vs 3%)
  - Higher risk per trade: 2.5% portfolio (vs 1.5%)

Uses v9's proven signal engine (ta-lib indicators, regime detection).
"""
from __future__ import annotations

import importlib, importlib.util, warnings, sys
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Load v9 module ────────────────────────────────────────────────────────────
_v9_path = Path(__file__).parent / "backtest_ta125_v9.py"
_spec = importlib.util.spec_from_file_location("v9", str(_v9_path))
v9 = importlib.util.module_from_spec(_spec)
sys.modules["v9"] = v9          # must be in sys.modules before exec (for @dataclass)
_spec.loader.exec_module(v9)

# Override module-level risk constants (used inside v9.run_backtest)
v9.RISK_PER_TRADE_PCT  = 0.025   # 2.5% portfolio at risk per trade (was 1.5%)
v9.MAX_SINGLE_POS_PCT  = 0.22    # 22% max per position (was 20%)

INDEX_TICKER = v9.INDEX_TICKER   # "^TA125.TA"
SIM_START    = v9.SIM_START      # 2021-04-01
SIM_END      = v9.SIM_END        # 2026-03-31
INITIAL_CASH = v9.INITIAL_CASH   # 100,000

# ── STSA strategy parameter variants ─────────────────────────────────────────
# Tuple format (same as v9):
#   (atr_mult, init_stop, tp_pct, min_bull_buy, min_bear_exit,
#    max_pos, pos_pct, min_hold, rs_min, mean_rev)

def _params(atr_mult=2.5, tp_bull=999.0, min_hold_bull=10,
            max_pos=5, pos_pct=0.22, rs_min=5.0):
    return {
        "BULL":    (atr_mult, 0.07, tp_bull, 5, 4,  max_pos, pos_pct, min_hold_bull, rs_min, False),
        "NEUTRAL": (1.5,      0.05, 0.20,    4, 3,  3,       0.10,    5,             0.0,    True),
        "BEAR":    (1.0,      0.04, 0.10,    9, 2,  0,       0.05,    0,             0.0,    False),
    }


STSA_VARIANTS = [
    # (label, params)
    ("STSA-A  Base (ATR=2.5x, hold>=10d, RS>=5%, pos=22%)",
     _params()),
    ("STSA-B  Wide trail (ATR=3.0x, hold>=10d)",
     _params(atr_mult=3.0)),
    ("STSA-C  Shorter hold (ATR=2.5x, hold>=5d, RS>=4%)",
     _params(min_hold_bull=5, rs_min=4.0)),
    ("STSA-D  More positions (max=7, pos=18%, RS>=4%)",
     _params(max_pos=7, pos_pct=0.18, rs_min=4.0)),
]

UNIVERSE = [
    "POLI.TA","LUMI.TA","DSCT.TA","MZTF.TA","FIBI.TA",
    "NICE.TA","CAMT.TA","TSEM.TA","NVMI.TA","ESLT.TA",
    "TEVA.TA","ICL.TA","BEZQ.TA","SKBN.TA","RSEL.TA",
    "PHOE.TA","HARL.TA","MGDL.TA",
    "AZRG.TA","AMOT.TA","ALHE.TA","ELCO.TA",
    "ENLT.TA","DLEKG.TA","ILCO.TA","SPEN.TA",
]


# ── Data download ─────────────────────────────────────────────────────────────
def _download(tickers, start, end, verbose=True):
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end,
                             auto_adjust=True, progress=False,
                             multi_level_index=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.capitalize() if c.lower() in
                          ("open","high","low","close","volume") else c
                          for c in df.columns]
            df.index = pd.DatetimeIndex([d.replace(tzinfo=None)
                                         if hasattr(d, "tzinfo") and d.tzinfo
                                         else d for d in df.index])
            df = df.dropna(subset=["Close","High","Low","Volume"])
            if len(df) >= 100:
                data[t] = df
                if verbose:
                    print(f"  {t}: {len(df)} days")
        except Exception as e:
            if verbose:
                print(f"  {t}: failed ({e})")
    return data


# ── Results formatter ─────────────────────────────────────────────────────────
def _fmt(r: dict, bh_return: float, bh_cagr: float, label: str):
    if r is None:
        print(f"  {label}: NO RESULT")
        return
    total     = r["total_ret"]
    cagr      = r["cagr"]
    wr        = r["win_rate"]
    n         = r["n_sells"]
    win_avg   = r["avg_win"]
    loss_avg  = r["avg_loss"]
    rr        = r["rr"]
    util      = r.get("avg_util", 0)
    stops     = r["stop_exits"]
    tps       = r["tp_exits"]
    sigs      = r["sig_exits"]
    bears     = r["bear_exits"]
    tax       = r.get("total_tax", 0)
    alpha     = total - bh_return
    cagr_alpha= cagr  - bh_cagr

    print(f"\n  {label}")
    print(f"    Return:      {total:+.2f}%   CAGR: {cagr:+.2f}%   Alpha vs index: {alpha:+.1f}%")
    print(f"    Final val:   NIS {r['total_val']:,.0f}   "
          f"Tax paid: NIS {tax:,.0f}")
    print(f"    Sharpe:      est. based on CAGR/drawdown (no daily data)")
    print(f"    Trades:      {n} sells   Win rate: {wr:.1f}%   R:R {rr:.2f}x")
    print(f"    Avg win:     NIS {win_avg:+.0f}   Avg loss: NIS {loss_avg:+.0f}")
    print(f"    Avg deployed:{util:.0f}%   Exits: {stops} trail | {tps} TP | {sigs} signal | {bears} bear")
    print(f"    vs TA-125:   return alpha {alpha:+.1f}%  CAGR alpha {cagr_alpha:+.2f}%")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sep = "=" * 72

    print(sep)
    print("  TASE STSA — SHORT-TERM SWING ALPHA STRATEGY")
    print("  Comparing concentrated short-hold vs v9 baseline vs TA-125 B&H")
    print(sep)

    # Fetch data
    print(f"\nFetching {len(UNIVERSE)} stocks...")
    all_data = _download(UNIVERSE, v9.FETCH_START, v9.FETCH_END, verbose=True)
    valid    = [t for t in UNIVERSE if t in all_data]
    print(f"Valid: {len(valid)} tickers\n")

    print(f"Fetching index {INDEX_TICKER}...")
    idx_raw = _download([INDEX_TICKER], v9.FETCH_START, v9.FETCH_END, verbose=False)
    if not idx_raw:
        raise RuntimeError("Index download failed.")
    index_df = list(idx_raw.values())[0]
    print(f"  {INDEX_TICKER}: {len(index_df)} days")

    # Benchmark
    sim_idx = index_df.loc[
        (index_df.index >= SIM_START) & (index_df.index <= SIM_END), "Close"
    ]
    n_years   = (SIM_END - SIM_START).days / 365.25
    bh_return = (sim_idx.iloc[-1] / sim_idx.iloc[0] - 1) * 100
    bh_cagr   = ((sim_idx.iloc[-1] / sim_idx.iloc[0]) ** (1/n_years) - 1) * 100

    print(f"\n{sep}")
    print(f"  BENCHMARK — TA-125 Buy & Hold")
    print(f"    NIS 100,000  ->  NIS {INITIAL_CASH * (1 + bh_return/100):,.0f}")
    print(f"    Return: {bh_return:+.2f}%   CAGR: {bh_cagr:+.2f}%")
    print(sep)

    # v9 AB baseline (reference from documented results)
    print("\n  V9 AB Baseline (documented, flat sizing, ATR=3.5x, hold>=40d):")
    print("    Return: +86.89%   CAGR: +13.32%   Win rate: ~54%   96 trail stops")
    print(f"    vs TA-125: {86.89 - bh_return:+.1f}% alpha")

    # Run STSA variants
    print(f"\n{sep}")
    print("  STSA VARIANTS (risk-parity sizing, all on same universe + index)")
    print(sep)

    results = {}
    for label, params in STSA_VARIANTS:
        print(f"\n  Running: {label}...")
        r = v9.run_backtest(valid, all_data, index_df, params,
                            use_risk_parity=True, label=label)
        results[label] = r
        _fmt(r, bh_return, bh_cagr, label)

    # Summary table
    print(f"\n{sep}")
    print("  SUMMARY TABLE")
    print(f"  {'Strategy':<50} {'Return':>8} {'CAGR':>7} {'Alpha':>8} {'WinRate':>8} {'Trades':>7}")
    print("  " + "-" * 68)

    rows = [
        ("TA-125 B&H (benchmark)",   bh_return, bh_cagr, 0, "-", "-"),
        ("V9 AB baseline (ref)",      86.89,     13.32,   86.89 - bh_return, 54, 96+10+20),
    ]
    for label, params in STSA_VARIANTS:
        r = results.get(label)
        if r:
            rows.append((label[:50], r["total_ret"], r["cagr"],
                         r["total_ret"] - bh_return, r["win_rate"], r["n_sells"]))

    for row in rows:
        name, ret, cagr, alpha, wr, n = row
        wr_s = f"{wr:.0f}%" if isinstance(wr, float) else str(wr)
        print(f"  {name:<50} {ret:>+7.1f}% {cagr:>+6.1f}% {alpha:>+7.1f}% {wr_s:>8} {str(n):>7}")

    print(sep)

    # Chart
    _plot(results, bh_return, bh_cagr, sim_idx)


def _plot(results: dict, bh_return, bh_cagr, bh_series):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 7))

        strategies = ["TA-125 B&H", "V9 AB"]
        returns    = [bh_return, 86.89]
        cagrs      = [bh_cagr, 13.32]
        colors     = ["orange", "green"]

        for label, params in STSA_VARIANTS:
            r = results.get(label)
            if r:
                strategies.append(label.split("  ")[0])
                returns.append(r["total_ret"])
                cagrs.append(r["cagr"])
                colors.append("royalblue")

        x = range(len(strategies))
        bars = ax.bar(x, returns, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

        # Label bars
        for bar, cagr in zip(bars, cagrs):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f"{h:+.0f}%\n(CAGR {cagr:.1f}%)",
                    ha="center", va="bottom", fontsize=8)

        ax.set_xticks(list(x))
        ax.set_xticklabels(strategies, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("5-Year Total Return (%)")
        ax.set_title("TASE Strategy Comparison — May 2021 to May 2026\n"
                     "Initial capital: NIS 100,000",
                     fontsize=11)
        ax.axhline(bh_return, color="orange", ls="--", lw=1.0,
                   label=f"TA-125 B&H ({bh_return:+.0f}%)")
        ax.axhline(0, color="black", lw=0.5)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        out = "backtest_stsa_comparison.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nChart saved -> {out}")

    except Exception as e:
        print(f"Chart error (non-fatal): {e}")


if __name__ == "__main__":
    main()
