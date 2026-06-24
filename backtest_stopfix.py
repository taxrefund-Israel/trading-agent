# -*- coding: utf-8 -*-
"""השוואת backtest מתחילת ההרצה עד היום: סטופ-לוס תמיד-פעיל (תוקן) מול חסום-ע"י-min_hold (ישן).
אותם נתונים, אותה אסטרטגיה — ההבדל היחיד הוא אכיפת הסטופ. רץ ב-DB מבודד (לא נוגע בתיק החי)."""
import warnings; warnings.filterwarnings("ignore")
import os, sys; sys.path.insert(0, ".")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import sqlite3
import db  # נשנה את נתיב ה-DB לבידוד

HERE = os.path.dirname(os.path.abspath(__file__))
START, ENDD = "2026-05-20", "2026-06-24"

def run_sim(stop_gated: bool, tag: str):
    tmp = os.path.join(HERE, f"_sim_{tag}.db")
    for ext in ("", "-wal", "-shm"):
        try: os.remove(tmp + ext)
        except OSError: pass
    db.DB_PATH = tmp
    import importlib, auto_trader
    importlib.reload(auto_trader)            # מאתחל מחדש עם ה-DB החדש
    auto_trader.STOP_LOSS_RESPECTS_MIN_HOLD = stop_gated
    auto_trader.init_db()
    days = auto_trader.get_tase_trading_days(START, ENDD)
    for d in days:
        auto_trader.run(for_date=d)
    conn = sqlite3.connect(tmp)
    val, cum, npos = conn.execute(
        "SELECT portfolio_value, cumulative_pct, open_positions FROM daily_snapshots ORDER BY date DESC LIMIT 1"
    ).fetchone()
    sells = conn.execute(
        "SELECT date(ts), symbol, pnl_pct, reason FROM trades WHERE action='SELL' ORDER BY ts"
    ).fetchall()
    conn.close()
    for ext in ("", "-wal", "-shm"):
        try: os.remove(tmp + ext)
        except OSError: pass
    return val, cum, npos, sells

print("מריץ סימולציה: סטופ חסום ע\"י min_hold (התנהגות ישנה)...")
v_old, c_old, n_old, s_old = run_sim(True, "old")
print("מריץ סימולציה: סטופ תמיד פעיל (מתוקן)...")
v_new, c_new, n_new, s_new = run_sim(False, "new")

print("\n" + "="*60)
print(f"{'תרחיש':28} {'שווי תיק':>12} {'תשואה':>9} {'פוז.':>5} {'מכירות':>7}")
print("-"*60)
print(f"{'ישן (סטופ חסום)':28} ₪{v_old:>10,.0f} {c_old:>+7.2f}% {n_old:>5} {len(s_old):>7}")
print(f"{'מתוקן (סטופ פעיל)':28} ₪{v_new:>10,.0f} {c_new:>+7.2f}% {n_new:>5} {len(s_new):>7}")
print("-"*60)
print(f"הפרש בתשואה: {c_new - c_old:+.2f} נק' אחוז  ({'לטובת המתוקן' if c_new>c_old else 'לטובת הישן'})")
print("="*60)
print("\nמכירות בתרחיש המתוקן (סטופים שנאכפו):")
for d, sym, pnl, r in s_new:
    print(f"  {d}  {sym:10} {pnl:+6.1f}%  {r}")
