"""
Scalper configuration — TASE intraday ORB + VWAP strategy.
Live execution via Interactive Brokers TWS API.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import os

# ─── Universe ─────────────────────────────────────────────────────────────────
# Most liquid TASE stocks — banks + large-cap tech/industrial
# Sorted by typical daily volume (highest first)
SCALP_UNIVERSE = [
    # Banks — most liquid on TASE
    "POLI.TA",   # Bank Hapoalim (POLI = Poalim)
    "LUMI.TA",   # Bank Leumi
    "MZTF.TA",   # Mizrahi-Tefahot Bank
    "DSCT.TA",   # Bank Discount
    "FIBI.TA",   # First International Bank
    # Large-cap industrials & tech
    "TEVA.TA",   # Teva Pharmaceutical
    "ICL.TA",    # ICL Group (minerals/chemicals)
    "BEZQ.TA",   # Bezeq telecom
    "NICE.TA",   # NICE Systems
    "ESLT.TA",   # Elbit Systems (defense)
    "TSEM.TA",   # Tower Semiconductor
    "NVMI.TA",   # Nova Measuring Instruments
    "CAMT.TA",   # Camtek (semiconductors)
    # Insurance & finance
    "PHOE.TA",   # Phoenix Holdings
    "MGDL.TA",   # Migdal Insurance
    "HARL.TA",   # Harel Insurance
    # Retail & real estate
    "SKBN.TA",   # Shufersal (supermarkets)
    "AZRG.TA",   # Azrieli Group (real estate)
    "AMOT.TA",   # Amot Investments (real estate)
    # Energy & materials
    "ENLT.TA",   # Enlight Renewable Energy
    "DLEKG.TA",  # Delek Group
    "RSEL.TA",   # Rishon LeZion (Electra Real Estate)
    "ALHE.TA",   # Al-Hal Electricity
    "ELCO.TA",   # Elco Holdings
    "ILCO.TA",   # Israel Corporation
    "ISCN.TA",   # Israel Canada (real estate)
    "SPEN.TA",   # Shapir Engineering
]

INDEX_TICKER = "^TA125.TA"

# ─── Market hours (Israel, UTC+3) ─────────────────────────────────────────────
MARKET_OPEN_H   = 9
MARKET_OPEN_M   = 59   # TASE opens 09:59
ORB_END_H       = 10
ORB_END_M       = 29   # Opening range = first 30 minutes
CLOSE_CUTOFF_H  = 16
CLOSE_CUTOFF_M  = 30   # Close all positions by 16:30 (45 min before close)
MARKET_CLOSE_H  = 17
MARKET_CLOSE_M  = 14

# ─── Strategy parameters ──────────────────────────────────────────────────────
@dataclass
class ScalperParams:
    # Entry / exit levels (relative to today's open price)
    orb_breakout_pct:    float = 0.002   # 0.2% above open = ORB trigger
    orb_stop_pct:        float = 0.004   # 0.4% below entry = hard stop
    target_risk_ratio:   float = 3.0     # target = 3× stop = 1.2% above entry

    # Filters
    min_volume_ratio:    float = 0.80    # skip if volume < 80% of 20d avg
    rsi_period:          int   = 14
    rsi_min_bull:        float = 45.0    # RSI > 45 for long signal
    atr_period:          int   = 14
    regime_sma:          int   = 50      # only long when index > SMA-50

    # Risk management
    risk_per_trade_pct:  float = 0.015   # 1.5% portfolio at risk per trade
    max_position_pct:    float = 0.30    # max 30% portfolio per stock
    max_open_positions:  int   = 3       # max simultaneous intraday positions
    max_daily_trades:    int   = 4       # limit to 4 trades/day (top signals only)

    # Costs — realistic for Interactive Brokers TASE
    # IB charges ~0.05% of trade value for Israeli stocks (min $2.50)
    commission_pct:      float = 0.0005  # 0.05% per side = 0.10% round-trip
    slippage_pct:        float = 0.0005  # 0.05% slippage (liquid bank stocks)
    tax_rate:            float = 0.25    # 25% Israeli capital gains (applied annually)

    # Backtest simulation
    orb_range_factor:    float = 0.45    # not used in new model (kept for live module)

# ─── Portfolio ────────────────────────────────────────────────────────────────
INITIAL_CASH = 100_000.0   # NIS

SIM_START   = "2021-05-01"   # 5 years (May 2021 -> May 2026), post-COVID
SIM_END     = "2026-05-01"
FETCH_START = "2020-01-01"   # extra year for indicator warmup (SMA200 + ATR)
WARMUP_DAYS = 300

# ─── Interactive Brokers connection ───────────────────────────────────────────
@dataclass
class IBConfig:
    host:      str = "127.0.0.1"
    port:      int = 7497          # 7497 = TWS paper trading; 7496 = live; 4002 = IB Gateway paper
    client_id: int = 10
    exchange:  str = "TASE"
    currency:  str = "ILS"
    timeout:   int = 20            # seconds to wait for IB connection

    # Bar size for real-time scalping
    bar_size:  str = "5 secs"      # 5-second bars — available in IB for real-time
    # For pre-market analysis: 1-minute historical bars
    hist_bar_size: str = "1 min"

ib_cfg = IBConfig()
params = ScalperParams()
