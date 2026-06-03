"""
SQLite database layer for the automated paper trading system.
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "paper_trades.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS positions (
        symbol          TEXT PRIMARY KEY,
        entry_price     REAL NOT NULL,
        entry_date      TEXT NOT NULL,
        qty             REAL NOT NULL,
        pos_pct         REAL NOT NULL,
        entry_regime    TEXT NOT NULL,
        trail_high      REAL NOT NULL,
        days_held       INTEGER DEFAULT 0,
        status          TEXT DEFAULT 'OPEN'  -- OPEN / CLOSED
    );

    CREATE TABLE IF NOT EXISTS trades (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ts              TEXT NOT NULL,          -- ISO timestamp
        symbol          TEXT NOT NULL,
        action          TEXT NOT NULL,          -- BUY / SELL
        price           REAL NOT NULL,
        qty             REAL NOT NULL,
        value           REAL NOT NULL,          -- price * qty
        reason          TEXT,                   -- trail_stop / signal_exit / take_profit / BUY
        pnl_pct         REAL,                   -- % gain/loss (on exits)
        pnl_nis         REAL,                   -- NIS gain/loss gross
        tax_nis         REAL,                   -- 25% on gains
        regime          TEXT
    );

    CREATE TABLE IF NOT EXISTS daily_snapshots (
        date            TEXT PRIMARY KEY,
        portfolio_value REAL NOT NULL,
        cash            REAL NOT NULL,
        bonds           REAL NOT NULL,
        open_positions  INTEGER NOT NULL,
        regime          TEXT NOT NULL,
        day_pnl         REAL,
        cumulative_pct  REAL,
        notes           TEXT
    );

    CREATE TABLE IF NOT EXISTS guardrail_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ts              TEXT NOT NULL,
        rule            TEXT NOT NULL,
        detail          TEXT,
        blocked         INTEGER DEFAULT 1       -- 1=blocked trade, 0=warning only
    );

    CREATE TABLE IF NOT EXISTS orders (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        created_ts      TEXT NOT NULL,          -- מתי נוצרה
        symbol          TEXT NOT NULL,
        action          TEXT NOT NULL,          -- BUY / SELL
        order_type      TEXT NOT NULL,          -- MARKET / LIMIT
        limit_price     REAL,                   -- מחיר לימיט (NULL=market)
        total_qty       REAL NOT NULL,          -- כמות מבוקשת
        filled_qty      REAL DEFAULT 0,         -- כמות שבוצעה
        filled_price    REAL,                   -- מחיר מילוי ממוצע
        filled_ts       TEXT,                   -- מתי בוצע
        status          TEXT DEFAULT 'PENDING', -- PENDING/FILLED/PARTIAL/EXPIRED/CANCELLED
        attempts        INTEGER DEFAULT 0,      -- כמה פעמים ניסינו
        max_attempts    INTEGER DEFAULT 3,      -- מקסימום ניסיונות
        tranche         INTEGER DEFAULT 1,      -- מספר טרנש (1/2/3)
        parent_id       INTEGER,                -- id של הפקודה המקורית (לטרנשים)
        regime          TEXT,
        reason          TEXT                    -- BUY_SIGNAL / TRAIL_STOP / etc.
    );
    """)
    conn.commit()
    conn.close()
    print(f"DB ready: {DB_PATH}")


if __name__ == "__main__":
    init_db()
