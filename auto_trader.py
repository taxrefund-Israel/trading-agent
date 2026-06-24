"""
auto_trader.py — Paper Trading Engine (v9-13 strategy)
Runs once daily after market close (17:30 Israel time).

Usage:
    python auto_trader.py
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import sys, os, math
from datetime import datetime, timedelta
import pandas as pd
import ta
import yfinance as yf

from db import init_db, get_conn
from guardrails import can_buy
from securities import BOND_INSTRUMENTS

# ─── Strategy config (v9-13) ──────────────────────────────────────────────────
INITIAL_CASH     = 100_000.0
COMMISSION       = 0.0008
TAX_RATE         = 0.25
RS_LOOKBACK      = 63
BOND_YIELD       = 0.038 / 252      # daily ~3.8% annual
RISK_PER_TRADE   = 0.015
MAX_POS_PCT      = 0.20

INDEX_TICKER     = "^TA125.TA"
FETCH_DAYS       = 420              # enough for SMA200

TA125_UNIVERSE = [
    "POLI.TA","LUMI.TA","DSCT.TA","FIBI.TA",
    "NICE.TA","CAMT.TA","TSEM.TA","NVMI.TA",
    "ESLT.TA","TEVA.TA","ICL.TA","BEZQ.TA",
    "SKBN.TA","RSEL.TA","HARL.TA","MGDL.TA",
    "AZRG.TA","AMOT.TA","ALHE.TA","ELCO.TA",
    "ENLT.TA","DLEKG.TA","ILCO.TA",
]

REGIME_PARAMS = {
    "BULL":    dict(atr_mult=3.5, init_stop=0.08, tp=999.0, max_pos=10,
                   pos_pct=0.12, min_hold=40, rs_min=3.0, mean_rev=False),
    "NEUTRAL": dict(atr_mult=2.0, init_stop=0.06, tp=0.30,  max_pos=4,
                   pos_pct=0.08, min_hold=20, rs_min=0.0, mean_rev=True),
    "BEAR":    dict(atr_mult=1.0, init_stop=0.05, tp=0.10,  max_pos=0,
                   pos_pct=0.05, min_hold=0,  rs_min=0.0, mean_rev=False),
}
BOND_ALLOC = {"BULL": 0.0, "NEUTRAL": 0.40, "BEAR": 0.60}

# ─── Ladder entry config ──────────────────────────────────────────────────────
# כל BUY מתפצל ל-3 טרנשים: 50% שוק + 30% ב-1%- + 20% ב-2%-
LADDER = [
    (1, 1.000, 0.50),   # tranche, price_factor, qty_fraction
    (2, 0.990, 0.30),
    (3, 0.980, 0.20),
]
LADDER_MAX_ATTEMPTS = 3   # ימים עד ביטול אוטומטי

# ─── Whole-units / minimum-lot config ──────────────────────────────────────────
# בבורסת ת"א נסחרות יחידות שלמות בלבד (אין שברי מניות).
# ברירת מחדל: יחידה אחת מינימום. ניתן להגדיר מינימום גבוה יותר לנייר ספציפי.
DEFAULT_MIN_QTY = 1
MIN_QTY = {
    # "SYM.TA": min_units,   # להוסיף כאן אם לנייר יש כמות מינימלית מיוחדת
}

def min_qty_for(sym):
    return MIN_QTY.get(sym, DEFAULT_MIN_QTY)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _last(s):
    v = s.dropna()
    return float(v.iloc[-1]) if len(v) else None

def compute_atr(close, high, low, w=14):
    s = ta.volatility.average_true_range(high, low, close, w)
    v = s.dropna()
    return float(v.iloc[-1]) if len(v) else None

def classify_regime(idx_df):
    c = idx_df["Close"]; h = idx_df["High"]; l = idx_df["Low"]
    sma200 = _last(ta.trend.sma_indicator(c, min(200, len(c)-1)))
    sma50s = ta.trend.sma_indicator(c, min(50, len(c)-1)).dropna()
    adx    = _last(ta.trend.adx(h, l, c, 14))
    cur    = float(c.iloc[-1])
    slope  = ((float(sma50s.iloc[-1]) - float(sma50s.iloc[-31])) / float(sma50s.iloc[-31])
              if len(sma50s) >= 31 else 0)
    if sma200 is None: return "NEUTRAL"
    if cur > sma200 and slope > 0.008 and (adx or 0) > 22: return "BULL"
    if cur < sma200 and slope < -0.008: return "BEAR"
    return "NEUTRAL"

def rs_score(df, idx_df):
    sh = df["Close"]; ih = idx_df["Close"]
    if len(sh) < RS_LOOKBACK+1 or len(ih) < RS_LOOKBACK+1: return None
    ih2 = ih[ih.index <= sh.index[-1]]
    if len(ih2) < RS_LOOKBACK+1: return None
    return ((float(sh.iloc[-1])/float(sh.iloc[-RS_LOOKBACK])-1) -
            (float(ih2.iloc[-1])/float(ih2.iloc[-RS_LOOKBACK])-1)) * 100

def buy_signal(df, idx_df, regime):
    p  = REGIME_PARAMS[regime]
    c  = df["Close"]; h = df["High"]; l = df["Low"]
    price = float(c.iloc[-1])
    sma20  = _last(ta.trend.sma_indicator(c, 20))
    sma50  = _last(ta.trend.sma_indicator(c, 50))
    rsi    = _last(ta.momentum.rsi(c, 14))
    bb     = ta.volatility.BollingerBands(c, 20, 2)
    bbl    = _last(bb.bollinger_lband()); bbu = _last(bb.bollinger_hband())
    bb_pct = (price - bbl)/(bbu - bbl) if bbu and bbl and bbu > bbl else None
    macd   = ta.trend.MACD(c, 12, 26, 9)
    ml = _last(macd.macd()); ms = _last(macd.macd_signal())
    n_bull = sum([
        bool(sma20 and price > sma20),
        bool(sma50 and price > sma50),
        bool(ml is not None and ms is not None and ml > ms),
        bool(rsi is not None and rsi < 55),
        bool(bb_pct is not None and bb_pct < 0.75),
    ])
    if regime == "BULL":
        rs = rs_score(df, idx_df)
        return n_bull >= 4 and (rs or 0) > p["rs_min"], n_bull
    elif regime == "NEUTRAL":
        ok = (rsi is not None and rsi < 38 and
              bb_pct is not None and bb_pct < 0.25)
        return ok, n_bull
    return False, 0

def risk_parity_qty(portfolio, price, atr, atr_mult, init_stop, cash):
    stop_dist = max(atr * atr_mult, price * init_stop) if atr else price * init_stop
    qty       = (portfolio * RISK_PER_TRADE) / stop_dist
    max_cap   = portfolio * MAX_POS_PCT / (price * (1 + COMMISSION))
    max_cash  = cash * 0.95 / (price * (1 + COMMISSION))
    raw       = max(0.0, min(qty, max_cap, max_cash))
    return int(math.floor(raw))   # יחידות שלמות בלבד — אין שברי מניות בת"א

def bear_exit_signals(df):
    c = df["Close"]; h = df["High"]; l = df["Low"]
    price  = float(c.iloc[-1])
    rsi    = _last(ta.momentum.rsi(c, 14))
    sma50  = _last(ta.trend.sma_indicator(c, 50))
    bb     = ta.volatility.BollingerBands(c, 20, 2)
    bbu    = _last(bb.bollinger_hband())
    macd   = ta.trend.MACD(c, 12, 26, 9)
    ml = _last(macd.macd()); ms = _last(macd.macd_signal())
    n = sum([
        bool(rsi and rsi > 65),
        bool(sma50 and price < sma50),
        bool(bbu and price > bbu),
        bool(ml is not None and ms is not None and ml < ms),
    ])
    return n

# ─── Load state from DB ───────────────────────────────────────────────────────
def load_state():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM positions WHERE status='OPEN'").fetchall()
    conn.close()
    positions = {r["symbol"]: dict(r) for r in rows}

    # Load cash & bonds from last snapshot
    conn = get_conn()
    snap = conn.execute(
        "SELECT * FROM daily_snapshots ORDER BY date DESC LIMIT 1"
    ).fetchone()
    conn.close()

    if snap:
        cash       = snap["cash"]
        bond_value = snap["bonds"]
    else:
        # First run — start fresh
        cash       = INITIAL_CASH
        bond_value = 0.0

    return positions, cash, bond_value

# ─── Save state to DB ─────────────────────────────────────────────────────────
def record_trade(sym, action, price, qty, reason, pnl_pct, pnl_nis, tax_nis, regime, trade_date=None):
    ts = trade_date or datetime.now().isoformat()
    conn = get_conn()
    conn.execute("""
        INSERT INTO trades (ts, symbol, action, price, qty, value, reason, pnl_pct, pnl_nis, tax_nis, regime)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (ts, sym, action, price, qty,
          price * qty, reason, pnl_pct, pnl_nis, tax_nis, regime))
    conn.commit()
    conn.close()

def save_position(pos):
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO positions
        (symbol, entry_price, entry_date, qty, pos_pct, entry_regime, trail_high, days_held, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (pos["symbol"], pos["entry_price"], pos["entry_date"],
          pos["qty"], pos["pos_pct"], pos["entry_regime"],
          pos["trail_high"], pos["days_held"], pos["status"]))
    conn.commit()
    conn.close()

def close_position(sym):
    conn = get_conn()
    conn.execute("UPDATE positions SET status='CLOSED' WHERE symbol=?", (sym,))
    conn.commit()
    conn.close()

def save_snapshot(date, portfolio_val, cash, bonds, open_pos, regime, initial_cap):
    day_pnl   = None
    cum_pct   = round((portfolio_val / initial_cap - 1) * 100, 2)
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO daily_snapshots
        (date, portfolio_value, cash, bonds, open_positions, regime, cumulative_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (date, portfolio_val, cash, bonds, open_pos, regime, cum_pct))
    conn.commit()
    conn.close()

# ─── Order management ────────────────────────────────────────────────────────
def create_ladder_orders(sym, base_price, total_qty, regime, reason, cash):
    """
    יוצר 3 פקודות לימיט (ladder) לכניסה מדורגת.
    מחזיר רשימת פקודות שנוצרו ואת המזומן שהוקצה.
    """
    now = datetime.now().isoformat()
    created = []
    reserved_cash = 0.0

    # פיצול ליחידות שלמות: כל טרנש מקבל יחידות שלמות, השארית לטרנש השוק (T1)
    total_qty = int(total_qty)
    allocs = [int(total_qty * frac) for _, _, frac in LADDER]
    allocs[0] += total_qty - sum(allocs)   # שארית → טרנש ראשון (MARKET)

    conn = get_conn()

    # פקודה ראשונה — parent_id = None תחילה
    parent_id = None
    for (tranche, price_factor, _qty_frac), qty in zip(LADDER, allocs):
        if qty <= 0:
            continue   # מדלגים על טרנש ריק (כשהכמות קטנה מדי לפיצול)

        limit_price = round(base_price * price_factor, 2)
        cost        = qty * limit_price * (1 + COMMISSION)

        if reserved_cash + cost > cash * 0.97:
            break   # אין מספיק מזומן לטרנש הזה

        order_type = "MARKET" if tranche == 1 else "LIMIT"
        cur = conn.execute("""
            INSERT INTO orders
            (created_ts, symbol, action, order_type, limit_price, total_qty,
             status, attempts, max_attempts, tranche, parent_id, regime, reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (now, sym, "BUY", order_type, limit_price, qty,
              "PENDING", 0, LADDER_MAX_ATTEMPTS, tranche,
              parent_id, regime, reason))

        if tranche == 1:
            parent_id = cur.lastrowid

        reserved_cash += cost
        created.append({
            "id": cur.lastrowid, "tranche": tranche,
            "limit_price": limit_price, "qty": qty,
        })

    conn.commit()
    conn.close()
    return created, reserved_cash


def try_fill_pending_orders(stock_data, cash, positions, regime, L, today_str=None):
    today_str = today_str or datetime.now().strftime("%Y-%m-%d")
    """
    מנסה למלא פקודות PENDING.
    לכל פקודה: בודק אם המחיר הגיע לרמת הלימיט היום.
    מחזיר cash מעודכן וסיכום פעולות.
    """
    conn = get_conn()
    pending = conn.execute(
        "SELECT * FROM orders WHERE status='PENDING' OR status='PARTIAL'"
    ).fetchall()
    conn.close()

    filled_syms = []
    for order in pending:
        sym = order["symbol"]
        if sym not in stock_data:
            continue

        df    = stock_data[sym]
        today_low  = float(df["Low"].iloc[-1])
        today_high = float(df["High"].iloc[-1])
        today_close= float(df["Close"].iloc[-1])

        order_id     = order["id"]
        limit_price  = order["limit_price"]
        qty          = order["total_qty"]
        order_type   = order["order_type"]
        attempts     = order["attempts"] + 1

        # בדיקת מילוי:
        # MARKET → תמיד מתמלא במחיר סגירה
        # LIMIT BUY → מתמלא אם LOW של היום <= limit_price
        can_fill = False
        fill_price = limit_price

        if order_type == "MARKET":
            can_fill   = True
            fill_price = today_close
        elif order_type == "LIMIT" and today_low <= limit_price:
            can_fill   = True
            fill_price = min(limit_price, today_close)  # fill at better price

        conn = get_conn()
        if can_fill:
            cost = qty * fill_price * (1 + COMMISSION)
            if cost > cash:
                # אין מספיק מזומן — ביטול
                conn.execute(
                    "UPDATE orders SET status='CANCELLED', attempts=? WHERE id=?",
                    (attempts, order_id))
                conn.commit()
                conn.close()
                L(f"    ORDER CANCELLED {sym} T{order['tranche']}: "
                  f"אין מספיק מזומן (₪{cash:,.0f} < ₪{cost:,.0f})")
            else:
                cash -= cost
                conn.execute("""
                    UPDATE orders SET status='FILLED', filled_qty=?, filled_price=?,
                    filled_ts=?, attempts=? WHERE id=?
                """, (qty, fill_price, datetime.now().isoformat(), attempts, order_id))
                conn.commit()
                conn.close()  # release lock before nested writes

                # עדכון פוזיציה קיימת (הוספת טרנש) או יצירה חדשה
                if sym in positions:
                    pos = positions[sym]
                    old_cost  = pos["avg_cost"] * pos["qty"]
                    new_qty   = pos["qty"] + qty
                    new_cost  = (old_cost + qty * fill_price) / new_qty
                    pos["qty"]      = new_qty
                    pos["avg_cost"] = new_cost
                    pos["trail_high"] = max(pos.get("trail_high", fill_price), fill_price)
                    save_position({**pos, "symbol": sym,
                                   "entry_price": pos["avg_cost"],
                                   "trail_high": pos["trail_high"]})
                else:
                    new_pos = {
                        "symbol": sym, "entry_price": fill_price,
                        "entry_date": today_str,
                        "qty": qty, "pos_pct": round(qty * fill_price / 100_000 * 100, 2),
                        "entry_regime": order["regime"],
                        "trail_high": fill_price, "days_held": 0, "status": "OPEN",
                        "avg_cost": fill_price,
                    }
                    positions[sym] = new_pos
                    save_position(new_pos)

                record_trade(sym, "BUY", fill_price, qty,
                             f"LADDER_T{order['tranche']}_FILLED",
                             None, None, None, regime, trade_date=today_str)

                tranche_label = f"T{order['tranche']}"
                L(f"    FILLED {sym} {tranche_label}: "
                  f"₪{fill_price:,.0f} x {qty:.2f} = ₪{cost:,.0f}")
                filled_syms.append(sym)

        else:
            # לא התמלא היום
            if attempts >= order["max_attempts"]:
                conn.execute(
                    "UPDATE orders SET status='EXPIRED', attempts=? WHERE id=?",
                    (attempts, order_id))
                L(f"    EXPIRED {sym} T{order['tranche']}: "
                  f"לא התמלא ב-{attempts} ניסיונות (לימיט ₪{limit_price:,.0f}, LOW ₪{today_low:,.0f})")
            else:
                conn.execute(
                    "UPDATE orders SET attempts=? WHERE id=?",
                    (attempts, order_id))
                L(f"    PENDING {sym} T{order['tranche']}: "
                  f"ניסיון {attempts}/{order['max_attempts']} "
                  f"(לימיט ₪{limit_price:,.0f}, LOW ₪{today_low:,.0f})")
            conn.commit()
            conn.close()

    return cash, filled_syms


# ─── Catchup helpers ─────────────────────────────────────────────────────────
def _snapshot_exists(date_str):
    conn = get_conn()
    row = conn.execute(
        "SELECT 1 FROM daily_snapshots WHERE date=?", (date_str,)
    ).fetchone()
    conn.close()
    return row is not None


def _snapshot_exists_any():
    conn = get_conn()
    row = conn.execute("SELECT 1 FROM daily_snapshots LIMIT 1").fetchone()
    conn.close()
    return row is not None


def get_tase_trading_days(start_str, end_str):
    """ימי המסחר בפועל בבורסת ת"א בין שני תאריכים (כולל), לפי נתוני מדד ת"א 125.
    מבוסס-נתונים בלבד — לא מניח ימים קבועים. מזהה אוטומטית את שבוע המסחר העדכני
    (כיום שני–שישי, כולל שישי בשעות מקוצרות; ראשון אינו יום מסחר) וגם חגי ישראל
    (כמו שבועות), כי yfinance מחזיר נר רק עבור ימי מסחר ממשיים. אם הבורסה תשנה שוב
    את לוח הימים — הזיהוי יתעדכן מאליו."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end   = datetime.strptime(end_str, "%Y-%m-%d") + timedelta(days=1)
    try:
        df = yf.Ticker(INDEX_TICKER).history(start=start, end=end)
        if df.empty:
            return []
        df.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in df.index])
        return [d.strftime("%Y-%m-%d") for d in df.index]
    except Exception:
        return []


def get_missed_trading_days():
    """ימי מסחר בת"א שאין להם snapshot, עד היום (לפי נתוני המדד בפועל)."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT MIN(date), MAX(date) FROM daily_snapshots"
    ).fetchone()
    existing = {r[0] for r in conn.execute("SELECT date FROM daily_snapshots").fetchall()}
    conn.close()

    if not rows or not rows[0]:
        return []  # no history at all

    first = rows[0]
    today = datetime.now().strftime("%Y-%m-%d")
    trading = get_tase_trading_days(first, today)
    # ימי מסחר אחרי ה-snapshot הראשון, שעדיין אין להם רישום
    return [d for d in trading if d > first and d not in existing]


def run_catchup(status_cb=None, notify=False):
    """Run auto-trader for every missed trading day in order.
    אם notify=True — נשלח מייל איתותים רק עבור היום האחרון (החדש ביותר)."""
    init_db()
    missed = get_missed_trading_days()
    if not missed:
        if status_cb: status_cb("אין ימים שהוחמצו")
        return []
    ran = []
    for day in missed:
        if status_cb: status_cb(f"מריץ {day}…")
        run(for_date=day, notify=(notify and day == missed[-1]))
        ran.append(day)
    return ran


# ─── Main run ─────────────────────────────────────────────────────────────────
def run(for_date=None, notify=False):
    today_str = for_date or datetime.now().strftime("%Y-%m-%d")
    log = []
    def L(msg): log.append(msg); print(msg)

    L(f"\n{'='*60}")
    L(f"  Auto Trader — {today_str}")
    L(f"{'='*60}")

    # Init DB if first run
    init_db()

    # Skip if already ran today
    if _snapshot_exists(today_str):
        L(f"  כבר רץ עבור {today_str} — דולג")
        return

    # Load state
    positions, cash, bond_value = load_state()
    L(f"  מצב קיים: {len(positions)} פוזיציות פתוחות, מזומן ₪{cash:,.0f}, אגח ₪{bond_value:,.0f}")

    # Fetch market data
    L("\n  טוען נתונים מ-yfinance...")
    end   = datetime.strptime(today_str, "%Y-%m-%d") + timedelta(days=1)
    start = end - timedelta(days=FETCH_DAYS)

    try:
        idx_df = yf.Ticker(INDEX_TICKER).history(start=start, end=end)
        idx_df.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in idx_df.index])
        if idx_df.empty or len(idx_df) < 60:
            L("  שגיאה: לא ניתן לטעון נתוני מדד. מפסיק.")
            return
    except Exception as e:
        L(f"  שגיאה בטעינת מדד: {e}")
        return

    # ── ודא שזהו יום מסחר בת"א (לא חג/סופ"ש) ─────────────────────────────────
    # אם הנר האחרון אינו של today_str, אין מסחר ביום הזה (חג כמו שבועות, או נתונים
    # שעוד לא התעדכנו) — לא יוצרים snapshot; ה-catchup יתפוס אותו כשהנתונים יגיעו.
    last_bar = idx_df.index[-1].strftime("%Y-%m-%d")
    if last_bar != today_str:
        L(f"  {today_str} אינו יום מסחר בת\"א (נר אחרון: {last_bar}) — דולג")
        return

    stock_data = {}
    skipped_no_bar = []
    for sym in TA125_UNIVERSE:
        try:
            df = yf.Ticker(sym).history(start=start - timedelta(days=50), end=end)
            if df.empty or len(df) < 60: continue
            df.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in df.index])
            # שמירה קריטית: סוחרים נייר רק אם נסחר בפועל ביום הזה (נר אמיתי ל-today_str).
            # כך לעולם לא נקנה/נמכור נייר שלא היה קיים/לא נסחר באותו יום (טרם הונפק,
            # הושעה, או נמחק מהמסחר).
            if df.index[-1].strftime("%Y-%m-%d") != today_str:
                skipped_no_bar.append(sym)
                continue
            stock_data[sym] = df
        except: pass
    L(f"  נטענו {len(stock_data)} מניות (נסחרו בפועל ב-{today_str})")
    if skipped_no_bar:
        L(f"  דולגו (אין נר מסחר ל-{today_str}): {', '.join(skipped_no_bar)}")

    # Regime
    regime = classify_regime(idx_df)
    p = REGIME_PARAMS[regime]
    L(f"\n  REGIME: {regime}")

    # Portfolio value (mark to market)
    equity = sum(
        pos["qty"] * float(stock_data[s]["Close"].iloc[-1])
        for s, pos in positions.items() if s in stock_data
    )

    # Bond daily yield
    bond_value *= (1 + BOND_YIELD)

    # Bond rebalance
    portfolio_val = cash + bond_value + equity
    target_bond = portfolio_val * BOND_ALLOC[regime]
    if bond_value < target_bond * 0.95 and cash > (target_bond - bond_value):
        move = min(target_bond - bond_value, cash * 0.4)
        bond_value += move; cash -= move
        L(f"  אגח: הוספה ₪{move:,.0f} (יעד {BOND_ALLOC[regime]*100:.0f}%)")
    elif bond_value > target_bond * 1.05:
        move = bond_value - target_bond
        cash += move; bond_value -= move
        L(f"  אגח: משיכה ₪{move:,.0f}")

    # ── Check exits ──────────────────────────────────────────────────────────
    L("\n  בודק יציאות...")
    exits = []
    for sym, pos in list(positions.items()):
        if sym not in stock_data: continue
        df  = stock_data[sym]
        price = float(df["Close"].iloc[-1])
        atr   = compute_atr(df["Close"], df["High"], df["Low"])

        # Update trail high & days held
        pos["trail_high"] = max(pos.get("trail_high", price), price)
        pos["days_held"]  = pos.get("days_held", 0) + 1

        trail_stop = max(
            pos["trail_high"] - atr * p["atr_mult"] if atr else pos["entry_price"] * 0.85,
            pos["entry_price"] * (1 - p["init_stop"])
        )
        tp_price = pos["entry_price"] * (1 + p["tp"]) if p["tp"] < 10 else None
        n_bear   = bear_exit_signals(df)

        exit_reason = None
        if price <= trail_stop:
            # סטופ-לוס/טריילינג — תמיד פעיל (בקרת סיכון!), ללא תלות בתקופת ההחזקה.
            # (לפני התיקון: היה חסום ע"י min_hold, כך שהסטופ לא נאכף ב-40 הימים הראשונים.)
            exit_reason = "trail_stop"
        elif tp_price and price >= tp_price:
            exit_reason = "take_profit"
        elif n_bear >= 4 and pos["days_held"] >= p["min_hold"]:
            # יציאת איתות (לא סטופ) — כאן min_hold כן רלוונטי, למניעת מכירה על רעש
            exit_reason = "signal_exit"
        elif regime == "BEAR" and pos["days_held"] >= 1:
            exit_reason = "regime_bear"

        if exit_reason:
            gross   = pos["qty"] * price * (1 - COMMISSION)
            cost    = pos["qty"] * pos["entry_price"]
            gain    = gross - cost
            tax     = max(gain * TAX_RATE, 0)
            net     = gross - tax
            pnl_pct = (price / pos["entry_price"] - 1) * 100
            cash   += net
            close_position(sym)
            record_trade(sym, "SELL", price, pos["qty"], exit_reason,
                         pnl_pct, gain, tax, regime, trade_date=today_str)
            # טווח מחיר מכירה סביב הסגירה האחרונה: יעד = מחיר שוק, רצפה מקובלת ~1%-
            exits.append({
                "sym": sym, "reason": exit_reason, "pnl_pct": pnl_pct,
                "qty": pos["qty"], "price": round(price, 2),
                "sell_low": round(price * 0.99, 2), "sell_high": round(price, 2),
            })
            L(f"    SELL {sym}: {pos['qty']:.0f} יח׳ @ ₪{price:,.0f} ({pnl_pct:+.1f}%) — {exit_reason}")
            del positions[sym]
        else:
            # Update position in DB
            save_position({
                "symbol": sym, "entry_price": pos["entry_price"],
                "entry_date": pos["entry_date"], "qty": pos["qty"],
                "pos_pct": pos["pos_pct"], "entry_regime": pos["entry_regime"],
                "trail_high": pos["trail_high"], "days_held": pos["days_held"],
                "status": "OPEN"
            })

    if not exits:
        L("    אין יציאות היום")

    # ── Fill pending orders first ────────────────────────────────────────────
    L("\n  בודק פקודות ממתינות (Ladder)...")
    cash, filled = try_fill_pending_orders(stock_data, cash, positions, regime, L, today_str)
    if not filled:
        L("    אין פקודות ממתינות")

    # ── Check new entries ────────────────────────────────────────────────────
    L("\n  בודק כניסות חדשות...")
    portfolio_val = cash + bond_value + sum(
        pos["qty"] * float(stock_data[s]["Close"].iloc[-1])
        for s, pos in positions.items() if s in stock_data
    )

    entries = []
    if regime != "BEAR":
        # סמלים שכבר יש פקודות פתוחות עליהם
        conn = get_conn()
        pending_syms = set(r["symbol"] for r in conn.execute(
            "SELECT DISTINCT symbol FROM orders WHERE status IN ('PENDING','PARTIAL')"
        ).fetchall())
        conn.close()

        candidates = []
        for sym, df in stock_data.items():
            if sym in positions or sym in pending_syms: continue
            if len(df) < 60: continue
            ok, n_bull = buy_signal(df, idx_df, regime)
            if ok:
                rs = rs_score(df, idx_df) or 0
                candidates.append((sym, rs, n_bull, df))
        candidates.sort(key=lambda x: x[1], reverse=True)

        slots = p["max_pos"] - len(positions) - len(pending_syms)
        for sym, rs, n_bull, df in candidates[:max(slots, 0)]:
            price = float(df["Close"].iloc[-1])
            atr   = compute_atr(df["Close"], df["High"], df["Low"])
            qty   = risk_parity_qty(portfolio_val, price, atr,
                                    p["atr_mult"], p["init_stop"], cash)
            pos_pct = min(qty * price / portfolio_val * 100, MAX_POS_PCT * 100)

            # יחידות שלמות + כמות מינימלית לרכישה
            min_q = min_qty_for(sym)
            if qty < min_q:
                L(f"    SKIP {sym}: כמות ({qty}) קטנה מהמינימום ({min_q}) או מהון זמין")
                continue

            allowed, block_reason = can_buy(sym, pos_pct, len(positions), portfolio_val)
            if not allowed:
                L(f"    BLOCKED {sym}: {block_reason}")
                continue

            # יצירת ladder orders (3 טרנשים)
            created, reserved = create_ladder_orders(
                sym, price, qty, regime, "BUY_SIGNAL", cash)

            if not created:
                L(f"    SKIP {sym}: אין מזומן לפקודות")
                continue

            # הקצאת מזומן לפקודות (שמור בצד, לא מוציאים עד מילוי)
            # הטרנש הראשון (MARKET) יתמלא בהרצה הבאה
            qty_total = sum(c["qty"] for c in created)
            entries.append({
                "sym": sym, "base_price": price, "pos_pct": round(pos_pct, 1),
                "rs": rs, "qty_total": qty_total,
                "tranches": [(c["tranche"], c["limit_price"], c["qty"]) for c in created],
            })
            L(f"    ORDER {sym}: ₪{price:,.0f}  |  {qty_total:.0f} יח'  |  {pos_pct:.1f}% תיק  "
              f"|  RS: {rs:+.1f}%  |  {len(created)} טרנשים נוצרו")

    if not entries:
        L("    אין כניסות חדשות היום")

    # ── Final snapshot ───────────────────────────────────────────────────────
    equity = sum(
        pos["qty"] * float(stock_data[s]["Close"].iloc[-1])
        for s, pos in positions.items() if s in stock_data
    )
    portfolio_val = cash + bond_value + equity
    cum_pct = (portfolio_val / INITIAL_CASH - 1) * 100
    save_snapshot(today_str, portfolio_val, cash, bond_value,
                  len(positions), regime, INITIAL_CASH)

    L(f"\n{'='*60}")
    L(f"  סיכום יומי — {today_str}")
    L(f"{'='*60}")
    L(f"  שווי תיק:      ₪{portfolio_val:,.0f}")
    L(f"  תשואה מצטברת: {cum_pct:+.2f}%")
    L(f"  מזומן:         ₪{cash:,.0f}")
    L(f"  אגח:           ₪{bond_value:,.0f}")
    L(f"  מניות (equity):₪{equity:,.0f}")
    L(f"  פוזיציות:      {len(positions)}")
    L(f"  Regime:        {regime}")
    L(f"  קניות היום:    {len(entries)}")
    L(f"  מכירות היום:   {len(exits)}")
    L(f"{'='*60}\n")

    # Save log to file
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"{today_str}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))

    # ── שליחת איתותים יומיים למייל (רק עבור ההרצה האחרונה/היום) ─────────────────
    if notify:
        try:
            from notifier import send_signals_email

            # החזקות נוכחיות (מניות)
            holdings_payload = []
            cur_prices = {}
            for s, pos in positions.items():
                cur = float(stock_data[s]["Close"].iloc[-1]) if s in stock_data else pos["entry_price"]
                cur_prices[s] = cur
                pnl = (cur / pos["entry_price"] - 1) * 100
                holdings_payload.append((s, pos["qty"], pos["entry_price"], cur, pnl))

            # ── תיק מומלץ לאחר ביצוע הפעולות ─────────────────────────────────────
            # מניחים שכל פקודות הקנייה של היום יתמלאו (במחיר הבסיס) — זהו תיק היעד.
            proj = {}  # sym -> {qty, value}
            for s, pos in positions.items():
                proj[s] = {"qty": pos["qty"], "value": pos["qty"] * cur_prices[s]}
            buy_cost = 0.0
            for e in entries:
                s = e["sym"]; q = e["qty_total"]; px = e["base_price"]
                buy_cost += q * px
                if s in proj:
                    proj[s]["qty"]   += q
                    proj[s]["value"] += q * px
                else:
                    proj[s] = {"qty": q, "value": q * px}
            proj_equity = sum(v["value"] for v in proj.values())
            proj_cash   = max(cash - buy_cost, 0.0)
            proj_total  = proj_equity + bond_value + proj_cash
            projected = {
                "total": proj_total, "equity": proj_equity,
                "bonds": bond_value, "cash": proj_cash,
                "stocks": [(s, d["qty"], d["value"]) for s, d in
                           sorted(proj.items(), key=lambda kv: kv[1]["value"], reverse=True)],
            }

            notify_payload = {
                "date": today_str, "regime": regime,
                "portfolio_val": portfolio_val, "cum_pct": cum_pct,
                "cash": cash, "bonds": bond_value, "equity": equity,
                "entries": entries, "exits": exits,
                "holdings": holdings_payload,
                "projected": projected,
                "bond_instruments": BOND_INSTRUMENTS.get(regime, BOND_INSTRUMENTS["NEUTRAL"]),
            }
            send_signals_email(notify_payload)        # מייל (אם מוגדר)
            try:
                from notifier_telegram import send_signals_telegram
                send_signals_telegram(notify_payload)  # טלגרם (אם מוגדר)
            except Exception as e:
                L(f"  [telegram] שגיאה: {e}")
        except Exception as e:
            L(f"  [notifier] שגיאה בשליחת התראות: {e}")


if __name__ == "__main__":
    init_db()
    # אם יש היסטוריה — השלם את כל ימי המסחר שהוחמצו (כולל היום, אם הנתונים זמינים).
    # אם אין היסטוריה כלל — הרצה ראשונית עבור היום.
    # notify=True → מייל איתותים נשלח עבור היום האחרון בלבד.
    if _snapshot_exists_any():
        run_catchup(status_cb=lambda m: print(f"  [catchup] {m}"), notify=True)
    else:
        run(notify=True)
