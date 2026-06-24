# -*- coding: utf-8 -*-
"""בודק לכל פוזיציה: רווח/הפסד, רמת הסטופ-לוס, והאם הסטופ נפרץ — וחשוב מכל,
האם יציאת הסטופ נחסמת ע"י תקופת ההחזקה המינימלית (min_hold)."""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from datetime import datetime, timedelta
import pandas as pd, yfinance as yf
from auto_trader import (classify_regime, compute_atr, REGIME_PARAMS, load_state,
    INDEX_TICKER, FETCH_DAYS)

end = datetime.today() + timedelta(days=1); start = end - timedelta(days=FETCH_DAYS)
idx = yf.Ticker(INDEX_TICKER).history(start=start, end=end)
idx.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in idx.index])
regime = classify_regime(idx); p = REGIME_PARAMS[regime]
positions, cash, bond = load_state()
print(f"REGIME={regime} | init_stop={p['init_stop']*100:.0f}% | atr_mult={p['atr_mult']} | min_hold={p['min_hold']}\n")
print(f"{'נייר':10} {'כניסה':>9} {'נוכחי':>9} {'P&L%':>7} {'סטופ':>9} {'נפרץ?':>6} {'ימים':>6} {'נמכר?':>22}")
print("-"*86)
for sym, pos in positions.items():
    try:
        df = yf.Ticker(sym).history(start=start - timedelta(days=50), end=end)
        df.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in df.index])
    except Exception:
        print(f"{sym}: no data"); continue
    price = float(df["Close"].iloc[-1]); entry = pos["entry_price"]
    atr = compute_atr(df["Close"], df["High"], df["Low"])
    dh = pos.get("days_held", 0) + 1
    pnl = (price/entry - 1) * 100
    init_stop_lvl = entry * (1 - p["init_stop"])
    trail = max((pos.get("trail_high", price) - atr*p["atr_mult"]) if atr else entry*0.85, init_stop_lvl)
    breached = price <= trail
    # הלוגיקה אחרי התיקון: סטופ-לוס תמיד פעיל (ללא min_hold)
    if breached:
        verdict = "כן — יימכר בהרצה הבאה ✅"
    else:
        verdict = "לא נפרץ"
    print(f"{sym:10} {entry:>9,.0f} {price:>9,.0f} {pnl:>+6.1f}% {trail:>9,.0f} "
          f"{'כן' if breached else 'לא':>6} {dh:>4}/{p['min_hold']} {verdict:>22}")
