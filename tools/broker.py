from __future__ import annotations
import json
from datetime import datetime
import yfinance as yf
from config import config
from tools.portfolio import record_trade, update_position_price


def _get_current_price(symbol: str) -> float | None:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        fast = ticker.fast_info
        return float(fast.last_price) if hasattr(fast, "last_price") else None
    except Exception:
        return None


def place_order(
    symbol: str,
    side: str,
    quantity: int,
    order_type: str = "market",
    limit_price: float | None = None,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> str:
    if not config.auto_execute:
        signal = {
            "type": "TRADE_SIGNAL",
            "status": "SIGNAL_ONLY",
            "symbol": symbol.upper(),
            "side": side.upper(),
            "quantity": quantity,
            "order_type": order_type,
            "limit_price": limit_price,
            "stop_loss_pct": stop_loss_pct or config.risk.stop_loss_pct,
            "take_profit_pct": take_profit_pct or config.risk.take_profit_pct,
            "generated_at": datetime.now().isoformat(),
            "message": (
                "AUTO_EXECUTE is disabled. Set AUTO_EXECUTE=true in .env to enable live trading. "
                "This signal has been logged but NOT executed."
            ),
        }
        # Log to signals file
        _log_signal(signal)
        return json.dumps(signal, indent=2)

    # Paper trading execution
    if config.paper_trading:
        return _paper_execute(symbol, side, quantity, order_type, limit_price, stop_loss_pct, take_profit_pct)

    # Live Alpaca execution
    return _alpaca_execute(symbol, side, quantity, order_type, limit_price, stop_loss_pct, take_profit_pct)


def _paper_execute(
    symbol: str, side: str, quantity: int, order_type: str,
    limit_price: float | None, stop_loss_pct: float | None, take_profit_pct: float | None
) -> str:
    price = _get_current_price(symbol)
    if price is None:
        return json.dumps({"error": f"Could not fetch current price for {symbol}"})

    execution_price = limit_price if (order_type == "limit" and limit_price) else price

    sl_pct = stop_loss_pct or config.risk.stop_loss_pct
    tp_pct = take_profit_pct or config.risk.take_profit_pct

    stop_loss = round(execution_price * (1 - sl_pct), 2) if side == "buy" else round(execution_price * (1 + sl_pct), 2)
    take_profit = round(execution_price * (1 + tp_pct), 2) if side == "buy" else round(execution_price * (1 - tp_pct), 2)

    trade = record_trade(symbol, side, quantity, execution_price, stop_loss, take_profit)
    if "error" in trade:
        return json.dumps(trade)

    return json.dumps({
        "status": "FILLED",
        "mode": "PAPER_TRADING",
        "symbol": symbol.upper(),
        "side": side.upper(),
        "quantity": quantity,
        "execution_price": round(execution_price, 2),
        "total_value": round(quantity * execution_price, 2),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "filled_at": datetime.now().isoformat(),
        "order_id": f"PAPER-{datetime.now().strftime('%Y%m%d%H%M%S')}",
    }, indent=2)


def _alpaca_execute(
    symbol: str, side: str, quantity: int, order_type: str,
    limit_price: float | None, stop_loss_pct: float | None, take_profit_pct: float | None
) -> str:
    if not config.alpaca_api_key or not config.alpaca_secret_key:
        return json.dumps({"error": "Alpaca API keys not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY."})

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        trading_client = TradingClient(
            config.alpaca_api_key,
            config.alpaca_secret_key,
            paper=True,
        )

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        if order_type == "limit" and limit_price:
            req = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )
        else:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )

        order = trading_client.submit_order(req)

        fill_price = float(order.filled_avg_price or limit_price or 0)
        if fill_price:
            sl_pct = stop_loss_pct or config.risk.stop_loss_pct
            tp_pct = take_profit_pct or config.risk.take_profit_pct
            stop_loss = round(fill_price * (1 - sl_pct), 2)
            take_profit = round(fill_price * (1 + tp_pct), 2)
            record_trade(symbol, side, quantity, fill_price, stop_loss, take_profit)

        return json.dumps({
            "status": str(order.status),
            "mode": "ALPACA_PAPER",
            "order_id": str(order.id),
            "symbol": symbol.upper(),
            "side": side.upper(),
            "quantity": quantity,
            "execution_price": fill_price,
            "filled_at": str(order.filled_at),
        }, indent=2)

    except ImportError:
        return json.dumps({"error": "alpaca-py not installed. Run: pip install alpaca-py"})
    except Exception as e:
        return json.dumps({"error": f"Alpaca execution failed: {e}"})


def get_positions() -> str:
    from tools.portfolio import get_portfolio_state
    return get_portfolio_state()


def _log_signal(signal: dict) -> None:
    try:
        import os
        log_file = "trading_signals.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(signal) + "\n")
    except Exception:
        pass
