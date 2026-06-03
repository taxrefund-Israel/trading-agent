from __future__ import annotations
import json
import yfinance as yf
from config import config
from tools.portfolio import _load


def check_risk_limits(symbol: str, side: str, quantity: int, price: float) -> str:
    state = _load()
    risk = config.risk

    rejects = []
    warnings = []

    cash = state["cash"]
    positions = state.get("positions", {})
    positions_value = sum(p["quantity"] * p["last_price"] for p in positions.values())
    total_value = cash + positions_value

    trade_value = quantity * price

    # 1. Sufficient cash
    if side == "buy" and trade_value > cash:
        rejects.append(f"Insufficient cash: need ${trade_value:,.2f}, have ${cash:,.2f}")

    # 2. Position size limit
    position_pct = trade_value / total_value
    if side == "buy" and position_pct > risk.max_position_pct:
        rejects.append(
            f"Position size {position_pct*100:.1f}% exceeds max {risk.max_position_pct*100:.0f}% "
            f"(max ${total_value * risk.max_position_pct:,.0f})"
        )
    elif side == "buy" and position_pct > risk.max_position_pct * 0.8:
        warnings.append(f"Position near size limit ({position_pct*100:.1f}%)")

    # 3. Max open positions
    current_positions = len(positions)
    if side == "buy" and symbol not in positions and current_positions >= risk.max_open_positions:
        rejects.append(f"Max open positions reached ({risk.max_open_positions})")

    # 4. Daily loss limit
    daily_start = state.get("daily_start_value", total_value)
    daily_pnl_pct = (total_value - daily_start) / daily_start * 100
    if daily_pnl_pct <= -(risk.max_daily_loss_pct * 100):
        rejects.append(
            f"Daily loss limit reached ({daily_pnl_pct:.1f}% — limit is {-risk.max_daily_loss_pct*100:.0f}%)"
        )
    elif daily_pnl_pct < -(risk.max_daily_loss_pct * 70):
        warnings.append(f"Approaching daily loss limit ({daily_pnl_pct:.1f}%)")

    # 5. Max drawdown (circuit breaker)
    peak = state.get("peak_value", total_value)
    drawdown_pct = (peak - total_value) / peak * 100
    if drawdown_pct >= risk.max_drawdown_pct * 100:
        rejects.append(
            f"Max drawdown circuit breaker triggered ({drawdown_pct:.1f}% — limit {risk.max_drawdown_pct*100:.0f}%)"
        )

    # 6. Selling without a position
    if side == "sell" and symbol not in positions:
        rejects.append(f"No existing position in {symbol} to sell")

    # 7. Selling more than held
    if side == "sell" and symbol in positions:
        held = positions[symbol]["quantity"]
        if quantity > held:
            rejects.append(f"Cannot sell {quantity} shares — only holding {held}")

    # 8. Liquidity check
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        avg_volume = info.get("averageVolume", 0)
        if avg_volume > 0 and quantity > avg_volume * 0.01:
            warnings.append(f"Order size ({quantity}) > 1% of avg daily volume ({avg_volume:,})")
    except Exception:
        pass

    # Calculate suggested position size
    max_allowed_value = total_value * risk.max_position_pct
    suggested_quantity = int(max_allowed_value / price) if price > 0 else 0
    stop_loss_price = round(price * (1 - risk.stop_loss_pct), 2)
    take_profit_price = round(price * (1 + risk.take_profit_pct), 2)
    risk_reward = risk.take_profit_pct / risk.stop_loss_pct

    if rejects:
        return json.dumps({
            "decision": "REJECTED",
            "reasons": rejects,
            "warnings": warnings,
            "portfolio_summary": {
                "total_value": round(total_value, 2),
                "daily_pnl_pct": round(daily_pnl_pct, 2),
                "drawdown_pct": round(drawdown_pct, 2),
                "open_positions": current_positions,
            },
        }, indent=2)

    return json.dumps({
        "decision": "APPROVED",
        "warnings": warnings,
        "trade_details": {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "trade_value": round(trade_value, 2),
            "position_pct_of_portfolio": round(position_pct * 100, 2),
            "suggested_stop_loss": stop_loss_price,
            "suggested_take_profit": take_profit_price,
            "risk_reward_ratio": round(risk_reward, 2),
        },
        "portfolio_after_trade": {
            "remaining_cash": round(cash - trade_value if side == "buy" else cash + trade_value, 2),
            "total_value": round(total_value, 2),
            "open_positions": current_positions + (1 if side == "buy" and symbol not in positions else 0),
        },
        "note": "Auto-execute: ENABLED" if config.auto_execute else "Auto-execute: DISABLED — requires confirmation",
    }, indent=2)
