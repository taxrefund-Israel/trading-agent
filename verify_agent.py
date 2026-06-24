# -*- coding: utf-8 -*-
"""בדיקת תקינות לסוכן ההשקעות: מתחקה אחרי החלטות הקנייה/מכירה במצב הנוכחי,
ומריץ בדיקות בקרה (תיק ריק / פוזיציה ותיקה) כדי להוכיח שהמנגנון אכן פועל."""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from datetime import datetime, timedelta
import pandas as pd, yfinance as yf
from auto_trader import (classify_regime, buy_signal, rs_score, REGIME_PARAMS,
    risk_parity_qty, compute_atr, bear_exit_signals, load_state, get_conn,
    INDEX_TICKER, TA125_UNIVERSE, FETCH_DAYS, INITIAL_CASH, MAX_POS_PCT, min_qty_for)
from guardrails import can_buy

end = datetime.today() + timedelta(days=1); start = end - timedelta(days=FETCH_DAYS)
idx = yf.Ticker(INDEX_TICKER).history(start=start, end=end)
idx.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in idx.index])
stock_data = {}
for sym in TA125_UNIVERSE:
    try:
        df = yf.Ticker(sym).history(start=start - timedelta(days=50), end=end)
        if df.empty or len(df) < 60: continue
        df.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in df.index])
        stock_data[sym] = df
    except Exception: pass

regime = classify_regime(idx); p = REGIME_PARAMS[regime]
positions, cash, bond = load_state()
pv = cash + bond + sum(pos["qty"] * float(stock_data[s]["Close"].iloc[-1])
                       for s, pos in positions.items() if s in stock_data)
print("="*64)
print(f"REGIME={regime} | max_pos={p['max_pos']} | min_hold={p['min_hold']} ימי מסחר")
print(f"positions={len(positions)} | cash=₪{cash:,.0f} | bond=₪{bond:,.0f} | portfolio=₪{pv:,.0f}")
print("="*64)

conn = get_conn()
pending = set(r["symbol"] for r in conn.execute(
    "SELECT DISTINCT symbol FROM orders WHERE status IN ('PENDING','PARTIAL')").fetchall())
conn.close()

print("\n--- בדיקת יציאות (פוזיציות מוחזקות) ---")
for sym, pos in positions.items():
    if sym not in stock_data:
        print(f"  {sym}: אין נתונים"); continue
    df = stock_data[sym]; price = float(df["Close"].iloc[-1])
    atr = compute_atr(df["Close"], df["High"], df["Low"]); entry = pos["entry_price"]
    dh = pos.get("days_held", 0) + 1
    trail = max((pos.get("trail_high", price) - atr * p["atr_mult"]) if atr else entry*0.85,
                entry * (1 - p["init_stop"]))
    tp = entry * (1 + p["tp"]) if p["tp"] < 10 else None
    nbear = bear_exit_signals(df)
    reason = None
    if price <= trail and dh >= p["min_hold"]: reason = "trail_stop"
    elif tp and price >= tp: reason = "take_profit"
    elif nbear >= 4 and dh >= p["min_hold"]: reason = "signal_exit"
    elif regime == "BEAR" and dh >= 1: reason = "regime_bear"
    gated = ((price <= trail or nbear >= 4) and dh < p["min_hold"])
    note = "  ← היה יוצא, אבל חסום ע\"י min_hold" if gated else ""
    print(f"  {sym}: ימים={dh}/{p['min_hold']} | מחיר={price:,.0f} trail={trail:,.0f} "
          f"nbear={nbear} | החלטה={reason or 'HOLD'}{note}")

print("\n--- בדיקת כניסות (מצב נוכחי) ---")
cands = []
for sym, df in stock_data.items():
    if sym in positions or sym in pending: continue
    ok, nb = buy_signal(df, idx, regime)
    if ok: cands.append((sym, rs_score(df, idx) or 0, nb))
cands.sort(key=lambda x: x[1], reverse=True)
slots = p["max_pos"] - len(positions) - len(pending)
print(f"  מועמדים שנתנו איתות קנייה: {[c[0] for c in cands] or 'אין'}")
print(f"  מקומות פנויים = max_pos({p['max_pos']}) − פוזיציות({len(positions)}) − ממתינות({len(pending)}) = {slots}")
print(f"  → ייקנו בפועל: {[c[0] for c in cands[:max(slots,0)]] or 'אין (אין מקום פנוי)'}")

print("\n--- בקרה: תיק ריק + מזומן מלא (להוכיח שהקנייה עובדת) ---")
tcash = INITIAL_CASH; slots2 = p["max_pos"]
would = 0
for sym, rs, nb in cands[:max(slots2, 0)]:
    df = stock_data[sym]; price = float(df["Close"].iloc[-1])
    atr = compute_atr(df["Close"], df["High"], df["Low"])
    qty = risk_parity_qty(tcash, price, atr, p["atr_mult"], p["init_stop"], tcash)
    allowed, br = can_buy(sym, min(qty*price/tcash*100, MAX_POS_PCT*100), would, tcash)
    ok_buy = qty >= min_qty_for(sym) and allowed
    if ok_buy: would += 1
    print(f"  {sym}: מחיר={price:,.0f} qty={qty} can_buy={allowed} → {'יקנה' if ok_buy else 'לא: '+(br if not allowed else 'qty=0')}")
print(f"  → בתיק ריק המנוע היה פותח {would} פוזיציות. {'✅ מנגנון הקנייה תקין' if would>0 else '(אין מועמדים כרגע)'}")
print("\n" + "="*64 + "\nבדיקה הושלמה.")
