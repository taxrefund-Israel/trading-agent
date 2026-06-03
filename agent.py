"""
Stock Technical Analysis & Trading Agent
Built with Claude Opus 4.7 + Anthropic SDK tool runner
"""
from __future__ import annotations
import json
import sys
import anthropic
from anthropic import beta_tool
from config import config
from tools.market_data import get_stock_data, get_market_news
from tools.indicators import calculate_indicators
from tools.portfolio import get_portfolio_state, get_trade_history
from tools.risk_manager import check_risk_limits
from tools.broker import place_order

client = anthropic.Anthropic(api_key=config.anthropic_api_key)

SYSTEM_PROMPT = f"""You are an expert stock trading agent specializing in technical analysis.

## Your Capabilities
- Fetch and analyze real-time and historical stock data
- Calculate and interpret technical indicators (RSI, MACD, Moving Averages, Bollinger Bands, Stochastic, ATR)
- Identify high-confidence trading opportunities through multi-indicator confluence
- Manage a portfolio within strict risk limits
- Execute or signal trades (based on AUTO_EXECUTE setting)

## Trading Philosophy
- **Confluence-based**: Require ≥{config.risk.min_signals_required} aligned signals before recommending a trade
- **Risk-first**: ALWAYS call check_risk() before execute_trade()
- **Trend-following**: Prefer trading in the direction of the dominant trend (SMA50/SMA200)
- **Confirmation**: Look for price action + momentum + volume alignment

## Risk Management Rules (NON-NEGOTIABLE)
- Max position size: {config.risk.max_position_pct*100:.0f}% of portfolio per stock
- Stop loss: {config.risk.stop_loss_pct*100:.0f}% below entry (default)
- Take profit: {config.risk.take_profit_pct*100:.0f}% above entry (default)
- Max daily loss: {config.risk.max_daily_loss_pct*100:.0f}% before halting
- Max drawdown circuit breaker: {config.risk.max_drawdown_pct*100:.0f}% from peak
- Max open positions: {config.risk.max_open_positions}

## Signal Scoring Framework
**Bullish signals (BUY bias)**:
- RSI(14) < 30 = oversold (strong signal)
- MACD bullish crossover
- Price above SMA20, SMA50, SMA200
- Golden Cross (SMA50 > SMA200)
- Price near/below lower Bollinger Band
- Stochastic < 20 (oversold)
- EMA12 > EMA26

**Bearish signals (SELL bias)**:
- RSI(14) > 70 = overbought (strong signal)
- MACD bearish crossover
- Price below SMA20, SMA50, SMA200
- Death Cross (SMA50 < SMA200)
- Price above upper Bollinger Band
- Stochastic > 80 (overbought)

## Workflow for Analysis
1. `analyze_technicals(symbol)` → get all indicators and initial signals
2. `fetch_stock_data(symbol)` → confirm price action and fundamentals
3. `get_news(symbol)` → news sentiment (optional, for confirmation)
4. `get_portfolio()` → check current portfolio state and limits
5. `check_risk(symbol, side, qty, price)` → MANDATORY before any trade
6. `execute_trade(...)` → only if risk check APPROVED and signals align

## Auto-Execute Status: {'ENABLED ⚡' if config.auto_execute else 'DISABLED 📊 (Signal-only mode)'}
{'All trades will be executed automatically in paper trading mode.' if config.auto_execute else 'Trades will be logged as signals only. Set AUTO_EXECUTE=true to enable execution.'}
"""


# ─── Tool definitions ──────────────────────────────────────────────────────────

@beta_tool
def analyze_technicals(symbol: str, period: str = "6mo") -> str:
    """Calculate comprehensive technical indicators and generate trading signals for a stock.
    Computes: SMA(20/50/200), EMA(12/26), RSI(14), MACD(12/26/9), Bollinger Bands(20,2),
    ATR(14), Stochastic(14,3), OBV. Returns indicator values, bullish/bearish signals,
    overall bias, and recommendation.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, NVDA, SPY).
        period: Historical period for calculation (1mo, 3mo, 6mo, 1y).
    """
    return calculate_indicators(symbol, period)


@beta_tool
def fetch_stock_data(symbol: str, period: str = "3mo", interval: str = "1d") -> str:
    """Fetch OHLCV stock data with key fundamentals (P/E, 52w high/low, market cap).

    Args:
        symbol: Stock ticker symbol.
        period: Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y.
        interval: Bar interval: 1m, 5m, 15m, 1h, 1d, 1wk.
    """
    return get_stock_data(symbol, period, interval)


@beta_tool
def get_news(symbol: str) -> str:
    """Get recent news headlines and sentiment analysis (positive/negative/neutral) for a stock.

    Args:
        symbol: Stock ticker symbol.
    """
    return get_market_news(symbol)


@beta_tool
def get_portfolio() -> str:
    """Get current portfolio: cash, open positions, unrealized P&L, daily performance,
    drawdown from peak, and risk limit status."""
    return get_portfolio_state()


@beta_tool
def get_history() -> str:
    """Get recent trade history (last 20 executed trades or signals)."""
    return get_trade_history(20)


@beta_tool
def check_risk(symbol: str, side: str, quantity: int, price: float) -> str:
    """Validate a proposed trade against all risk management rules BEFORE executing.
    Checks position size, daily loss limit, drawdown circuit breaker, cash availability.
    Returns APPROVED or REJECTED with specific reasons and suggested parameters.

    Args:
        symbol: Stock ticker symbol.
        side: Trade direction — 'buy' or 'sell'.
        quantity: Number of shares to trade.
        price: Expected execution price per share.
    """
    return check_risk_limits(symbol, side, quantity, price)


@beta_tool
def execute_trade(
    symbol: str,
    side: str,
    quantity: int,
    order_type: str = "market",
    limit_price: float | None = None,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.15,
) -> str:
    """Execute a trade order (or log as a signal if AUTO_EXECUTE=false).
    MANDATORY: Call check_risk() first and only proceed if decision is APPROVED.

    Args:
        symbol: Stock ticker symbol.
        side: 'buy' or 'sell'.
        quantity: Number of shares.
        order_type: 'market' (default) or 'limit'.
        limit_price: Limit price, required only when order_type is 'limit'.
        stop_loss_pct: Stop loss percentage below entry price (default 0.05 = 5%).
        take_profit_pct: Take profit percentage above entry price (default 0.15 = 15%).
    """
    return place_order(symbol, side, quantity, order_type, limit_price, stop_loss_pct, take_profit_pct)


@beta_tool
def scan_watchlist(symbols: list[str] | None = None) -> str:
    """Scan multiple stocks and rank by technical signal strength.
    Returns a summary of each stock's bias (BULLISH/BEARISH/NEUTRAL) and signal count.

    Args:
        symbols: List of tickers to scan. Uses default watchlist if not provided.
    """
    targets = symbols or config.default_watchlist
    results = []
    for sym in targets[:12]:
        try:
            data = json.loads(calculate_indicators(sym, "3mo"))
            if "error" not in data:
                results.append({
                    "symbol": sym,
                    "last_close": data.get("last_close"),
                    "change_1d_pct": data.get("change_1d_pct"),
                    "bias": data.get("overall_bias"),
                    "bullish_signals": data["signal_count"]["bullish"],
                    "bearish_signals": data["signal_count"]["bearish"],
                    "recommendation": data.get("recommendation"),
                    "rsi": data["indicators"].get("RSI_14"),
                })
            else:
                results.append({"symbol": sym, "error": data["error"]})
        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})

    results.sort(key=lambda x: x.get("bullish_signals", 0) - x.get("bearish_signals", 0), reverse=True)
    return json.dumps({"scan_results": results, "scanned_at": __import__("datetime").datetime.now().isoformat()}, indent=2)


# ─── Agent runner ──────────────────────────────────────────────────────────────

TOOLS = [analyze_technicals, fetch_stock_data, get_news, get_portfolio, get_history,
         check_risk, execute_trade, scan_watchlist]


def run(user_message: str, stream_output: bool = True) -> str:
    """Run the trading agent for a single query."""
    print(f"\n{'='*60}")
    print(f"🤖 Trading Agent")
    print(f"{'='*60}")
    print(f"Query: {user_message}\n")

    if stream_output:
        return _run_streaming(user_message)
    else:
        return _run_blocking(user_message)


def _run_streaming(user_message: str) -> str:
    full_response = ""
    tool_calls_made = []

    runner = client.beta.messages.tool_runner(
        model=config.model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=[{"role": "user", "content": user_message}],
    )

    for message in runner:
        for block in message.content:
            if block.type == "text":
                print(block.text)
                full_response = block.text
            elif block.type == "tool_use":
                tool_calls_made.append(block.name)
                print(f"\n📊 [{block.name}] called...")

    print(f"\n{'─'*60}")
    if tool_calls_made:
        print(f"Tools used: {', '.join(tool_calls_made)}")
    return full_response


def _run_blocking(user_message: str) -> str:
    runner = client.beta.messages.tool_runner(
        model=config.model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=[{"role": "user", "content": user_message}],
    )

    final_text = ""
    for message in runner:
        for block in message.content:
            if block.type == "text":
                final_text = block.text
    return final_text


def interactive():
    """Interactive CLI session."""
    print("\n" + "="*60)
    print("  📈 Stock Technical Analysis & Trading Agent")
    print("  Powered by Claude Opus 4.7")
    print("="*60)
    print(f"\n  Mode: {'🟢 AUTO-EXECUTE (Paper Trading)' if config.auto_execute else '📊 SIGNAL-ONLY'}")
    print(f"  Watchlist: {', '.join(config.default_watchlist)}")
    print("\n  Example queries:")
    print("  • 'Analyze AAPL and tell me if I should buy'")
    print("  • 'Scan my watchlist for opportunities'")
    print("  • 'What does my portfolio look like?'")
    print("  • 'Show me the technical setup for NVDA and TSLA'")
    print("  • 'Find the best buy signal from the watchlist and execute it'")
    print("\n  Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye! 👋")
                break
            run(user_input)
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run(query)
    else:
        interactive()
