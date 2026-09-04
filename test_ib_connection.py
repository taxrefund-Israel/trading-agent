# -*- coding: utf-8 -*-
"""
בדיקת התחברות ל-IBKR (חשבון דמו) — לא שולח שום פקודה.
הרצה: python test_ib_connection.py
דרישות: IB Gateway או TWS פתוח ומחובר לחשבון ה-Free Trial,
         API מאופשר (Global Config -> API -> Settings -> Enable ActiveX and Socket Clients),
         פורט 7497 (TWS) או 4002 (Gateway) — עדכן למטה אם שינית.
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from ib_async import IB, Stock

HOST, PORTS = "127.0.0.1", [7497, 4002]

ib = IB()
connected = False
for port in PORTS:
    try:
        print(f"מנסה להתחבר אל {HOST}:{port} ...")
        ib.connect(HOST, port, clientId=99, timeout=10)
        connected = True
        print(f"✓ מחובר בפורט {port}")
        break
    except Exception as e:
        print(f"  לא הצליח ({type(e).__name__})")

if not connected:
    print("\n✗ אין חיבור. ודא ש-TWS/Gateway פתוח, מחובר לחשבון, וה-API מאופשר.")
    sys.exit(1)

accounts = ib.managedAccounts()
paper = all(a.startswith(("DU", "DF")) for a in accounts)
print(f"חשבונות: {accounts}  ({'דמו ✓' if paper else 'אמיתי! זהירות'})")

summary = {r.tag: r.value for r in ib.accountSummary()
           if r.tag in ("NetLiquidation", "TotalCashValue", "BuyingPower")}
for k, v in summary.items():
    print(f"  {k}: {float(v):,.0f}")

print("\nבדיקת נתוני שוק (AAPL, delayed):")
ib.reqMarketDataType(3)
c = Stock("AAPL", "SMART", "USD")
ib.qualifyContracts(c)
t = ib.reqMktData(c, "", False, False)
ib.sleep(3)
print(f"  AAPL last/close: {t.last} / {t.close}")

ib.disconnect()
print("\n✓ הכול תקין — אפשר להריץ את us_executor.py")
