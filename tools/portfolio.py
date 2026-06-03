from __future__ import annotations
import json
import os
from datetime import datetime, date
from config import config


def _load() -> dict:
    if os.path.exists(config.portfolio_file):
        with open(config.portfolio_file) as f:
            return json.load(f)
    return {
        "cash": config.initial_cash,
        "peak_value": config.initial_cash,
        "daily_start_value": config.initial_cash,
        "daily_start_date": str(date.today()),
        "positions": {},
        "trades": [],
        "created_at": datetime.now().isoformat(),
    }


def _save(state: dict) -> None:
    with open(config.portfolio_file, "w") as f:
        json.dump(state, f, indent=2, default=str)


def get_portfolio_state() -> str:
    state = _load()
    _refresh_daily_baseline(state)

    positions = state.get("positions", {})
    cash = state["cash"]

    # Calculate current market value using last known prices
    positions_value = sum(
        p["quantity"] * p["last_price"] for p in positions.values()
    )
    total_value = cash + positions_value
    daily_pnl = total_value - state["daily_start_value"]
    daily_pnl_pct = (daily_pnl / state["daily_start_value"]) * 100
    drawdown_pct = ((state["peak_value"] - total_value) / state["peak_value"]) * 100

    result = {
        "cash": round(cash, 2),
        "positions_value": round(positions_value, 2),
        "total_portfolio_value": round(total_value, 2),
        "daily_pnl": round(daily_pnl, 2),
        "daily_pnl_pct": round(daily_pnl_pct, 2),
        "drawdown_from_peak_pct": round(drawdown_pct, 2),
        "peak_value": round(state["peak_value"], 2),
        "open_positions": len(positions),
        "positions": {
            sym: {
                "quantity": p["quantity"],
                "avg_cost": round(p["avg_cost"], 2),
                "last_price": round(p["last_price"], 2),
                "market_value": round(p["quantity"] * p["last_price"], 2),
                "unrealized_pnl": round((p["last_price"] - p["avg_cost"]) * p["quantity"], 2),
                "unrealized_pnl_pct": round((p["last_price"] - p["avg_cost"]) / p["avg_cost"] * 100, 2),
                "stop_loss": p.get("stop_loss"),
                "take_profit": p.get("take_profit"),
            }
            for sym, p in positions.items()
        },
        "risk_limits": {
            "max_position_pct": config.risk.max_position_pct * 100,
            "max_daily_loss_pct": config.risk.max_daily_loss_pct * 100,
            "max_drawdown_pct": config.risk.max_drawdown_pct * 100,
            "auto_execute_enabled": config.auto_execute,
        },
    }
    return json.dumps(result, indent=2)


def get_trade_history(limit: int = 20) -> str:
    state = _load()
    trades = state.get("trades", [])[-limit:]
    return json.dumps({"trades": list(reversed(trades)), "total_trades": len(state.get("trades", []))}, indent=2)


def record_trade(symbol: str, side: str, quantity: int, price: float,
                 stop_loss: float | None, take_profit: float | None) -> dict:
    state = _load()
    positions = state.setdefault("positions", {})
    cost = quantity * price

    if side == "buy":
        if symbol in positions:
            pos = positions[symbol]
            total_qty = pos["quantity"] + quantity
            pos["avg_cost"] = (pos["avg_cost"] * pos["quantity"] + cost) / total_qty
            pos["quantity"] = total_qty
        else:
            positions[symbol] = {
                "quantity": quantity,
                "avg_cost": price,
                "last_price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "opened_at": datetime.now().isoformat(),
            }
        state["cash"] -= cost

    elif side == "sell":
        if symbol not in positions:
            return {"error": f"No position in {symbol}"}
        pos = positions[symbol]
        sell_qty = min(quantity, pos["quantity"])
        realized_pnl = (price - pos["avg_cost"]) * sell_qty
        pos["quantity"] -= sell_qty
        if pos["quantity"] <= 0:
            del positions[symbol]
        state["cash"] += sell_qty * price

    # Track peak
    positions_value = sum(p["quantity"] * p["last_price"] for p in positions.values())
    total_value = state["cash"] + positions_value
    if total_value > state["peak_value"]:
        state["peak_value"] = total_value

    trade = {
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": price,
        "value": round(quantity * price, 2),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "timestamp": datetime.now().isoformat(),
    }
    state.setdefault("trades", []).append(trade)
    _save(state)
    return trade


def update_position_price(symbol: str, price: float) -> None:
    state = _load()
    if symbol in state.get("positions", {}):
        state["positions"][symbol]["last_price"] = price
        _save(state)


def _refresh_daily_baseline(state: dict) -> None:
    today = str(date.today())
    if state.get("daily_start_date") != today:
        positions_value = sum(p["quantity"] * p["last_price"] for p in state.get("positions", {}).values())
        state["daily_start_value"] = state["cash"] + positions_value
        state["daily_start_date"] = today
        _save(state)
