"""
Hard guardrails — the AI never overrides these.
All rules are checked before any trade is executed.
"""
from datetime import datetime
from db import get_conn

# ─── Risk limits ───────────────────────────────────────────────────────────────
MAX_SINGLE_POSITION_PCT = 0.20      # max 20% of portfolio in one stock
MAX_OPEN_POSITIONS      = 10        # max concurrent positions
MAX_DAILY_BUYS          = 4         # max new BUY orders per day
MAX_PORTFOLIO_DRAWDOWN  = 0.15      # stop all trading if portfolio drops 15% from peak
INITIAL_CAPITAL         = 100_000.0 # used to compute drawdown


def _log(rule, detail, blocked=True):
    conn = get_conn()
    conn.execute(
        "INSERT INTO guardrail_log (ts, rule, detail, blocked) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(), rule, detail, int(blocked))
    )
    conn.commit()
    conn.close()


def is_trading_day():
    """Israeli market: Sun–Thu only."""
    wd = datetime.today().weekday()   # Mon=0 … Sun=6
    if wd in (4, 5):  # Fri=4, Sat=5
        _log("TRADING_DAY", f"Market closed today (weekday={wd})", blocked=True)
        return False
    return True


def check_drawdown(portfolio_value):
    """Block all trading if portfolio dropped >15% from peak recorded in DB."""
    conn = get_conn()
    row = conn.execute(
        "SELECT MAX(portfolio_value) as peak FROM daily_snapshots"
    ).fetchone()
    conn.close()
    peak = row["peak"] if row["peak"] else INITIAL_CAPITAL
    drawdown = (peak - portfolio_value) / peak
    if drawdown >= MAX_PORTFOLIO_DRAWDOWN:
        _log("DRAWDOWN", f"Portfolio ₪{portfolio_value:,.0f} is {drawdown*100:.1f}% below peak ₪{peak:,.0f}")
        return False, drawdown
    return True, drawdown


def check_position_size(pos_pct):
    if pos_pct > MAX_SINGLE_POSITION_PCT * 100:
        _log("POSITION_SIZE", f"Requested {pos_pct:.1f}% > max {MAX_SINGLE_POSITION_PCT*100:.0f}%")
        return False
    return True


def check_max_positions(current_open):
    if current_open >= MAX_OPEN_POSITIONS:
        _log("MAX_POSITIONS", f"Already {current_open} open positions (max {MAX_OPEN_POSITIONS})", blocked=True)
        return False
    return True


def check_daily_buys():
    """Count BUY trades made today."""
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_conn()
    count = conn.execute(
        "SELECT COUNT(*) as n FROM trades WHERE action='BUY' AND ts LIKE ?",
        (f"{today}%",)
    ).fetchone()["n"]
    conn.close()
    if count >= MAX_DAILY_BUYS:
        _log("DAILY_BUY_LIMIT", f"Already {count} buys today (max {MAX_DAILY_BUYS})", blocked=True)
        return False
    return True


def can_buy(sym, pos_pct, open_positions, portfolio_value):
    """Run all pre-trade checks. Returns (allowed: bool, reason: str)."""
    if not is_trading_day():
        return False, "שוק סגור היום (שישי/שבת)"
    ok, dd = check_drawdown(portfolio_value)
    if not ok:
        return False, f"DRAWDOWN GUARD: ירידה של {dd*100:.1f}% מהשיא — מסחר מושהה"
    if not check_max_positions(open_positions):
        return False, f"יותר מדי פוזיציות פתוחות ({open_positions})"
    if not check_daily_buys():
        return False, "הגעת למגבלת הקניות היומית"
    if not check_position_size(pos_pct):
        return False, f"גודל פוזיציה ({pos_pct:.1f}%) חורג מהמגבלה"
    return True, "OK"
