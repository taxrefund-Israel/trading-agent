from __future__ import annotations
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RiskLimits:
    max_position_pct: float = float(os.getenv("MAX_POSITION_PCT", "0.10"))
    max_daily_loss_pct: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05"))
    max_drawdown_pct: float = float(os.getenv("MAX_DRAWDOWN_PCT", "0.15"))
    stop_loss_pct: float = float(os.getenv("STOP_LOSS_PCT", "0.05"))
    take_profit_pct: float = float(os.getenv("TAKE_PROFIT_PCT", "0.15"))
    max_open_positions: int = int(os.getenv("MAX_OPEN_POSITIONS", "10"))
    min_signals_required: int = 3


@dataclass
class AgentConfig:
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    alpaca_api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    paper_trading: bool = True
    auto_execute: bool = field(default_factory=lambda: os.getenv("AUTO_EXECUTE", "false").lower() == "true")
    initial_cash: float = float(os.getenv("INITIAL_CASH", "100000"))
    model: str = "claude-opus-4-7"
    risk: RiskLimits = field(default_factory=RiskLimits)
    default_watchlist: list = field(default_factory=lambda: [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "SPY", "QQQ", "AMD"
    ])
    portfolio_file: str = "portfolio_state.json"


config = AgentConfig()
