# -*- coding: utf-8 -*-
"""
IBKR Sync — קורא את המצב האמיתי מחשבון IBKR (דמו/אמיתי) וכותב
ibkr_snapshot.json, ואז דוחף לריפו כדי שהדשבורד בענן יציג תוצאות בפועל.

הרצה: python us_ibkr_sync.py           (סנכרון + push)
       python us_ibkr_sync.py --no-push (מקומי בלבד)
מתוזמן: מומלץ פעם ביום אחרי סגירת המסחר (23:30 IL) — משימת USIBKRSync_Daily.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(BASE, "ibkr_snapshot.json")
STREAK = os.path.join(BASE, "logs_us", "sync_fail_streak.json")
ALERT_AFTER = 3   # התראה החל מ-3 כשלונות רצופים, וכל 3 נוספים


def _tg_send(text: str) -> None:
    import urllib.parse
    import urllib.request
    cfg_path = os.path.join(BASE, "telegram_config.json")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat) and os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            c = json.load(f)
        token, chat = c.get("bot_token"), c.get("chat_id")
    if not (token and chat):
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text,
                                   "parse_mode": "HTML"}).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage", data=data)
    try:
        urllib.request.urlopen(req, timeout=30)
    except Exception as e:
        print(f"[telegram] שליחה נכשלה: {e}")


def _load_streak() -> int:
    try:
        with open(STREAK, encoding="utf-8") as f:
            return int(json.load(f).get("streak", 0))
    except Exception:
        return 0


def _save_streak(n: int) -> None:
    os.makedirs(os.path.dirname(STREAK), exist_ok=True)
    with open(STREAK, "w", encoding="utf-8") as f:
        json.dump({"streak": n, "updated": datetime.now().isoformat()}, f)


def _on_failure() -> None:
    streak = _load_streak() + 1
    _save_streak(streak)
    print(f"אין חיבור ל-IB Gateway — כשלון מס' {streak} ברצף.")
    if streak >= ALERT_AFTER and streak % ALERT_AFTER == 0:
        last = "—"
        try:
            with open(SNAP, encoding="utf-8") as f:
                last = json.load(f).get("updated", "—")
        except Exception:
            pass
        _tg_send(f'🟠 <b>הסנכרון הלילי מ-IBKR נכשל {streak} לילות ברצף</b>\n'
                 f'ה-Gateway לא היה מחובר. הדשבורד מציג נתונים מ-{last}.\n'
                 f'פתח את IB Gateway והתחבר — הסנכרון הבא (23:30) יתעדכן אוטומטית.')


def _on_success() -> None:
    streak = _load_streak()
    if streak >= ALERT_AFTER:
        _tg_send(f'🟢 הסנכרון הלילי מ-IBKR חזר לעבוד (אחרי {streak} כשלונות רצופים). '
                 f'הדשבורד מעודכן.')
    if streak:
        _save_streak(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    from ib_async import IB
    ib = IB()
    for port in (4002, 7497):
        try:
            ib.connect("127.0.0.1", port, clientId=22, timeout=12)
            break
        except Exception:
            continue
    if not ib.isConnected():
        _on_failure()
        sys.exit(1)

    accounts = ib.managedAccounts()
    netliq = cash = None
    for v in ib.accountValues():
        if v.currency != "USD":
            continue
        if v.tag == "NetLiquidation":
            netliq = float(v.value)
        elif v.tag == "TotalCashValue":
            cash = float(v.value)

    positions = []
    ib.reqMarketDataType(3)
    for p in ib.positions():
        if p.position == 0:
            continue
        sym = p.contract.symbol.replace(" ", "-")
        avg = float(p.avgCost)
        # מחיר שוק: דרך פורטפוליו (כולל delayed)
        positions.append({"sym": sym, "qty": int(p.position), "avg_cost": round(avg, 2)})
    # שווי ורווח מה-portfolio items (יש שם marketPrice)
    port_items = {i.contract.symbol.replace(" ", "-"): i for i in ib.portfolio()}
    for pos in positions:
        it = port_items.get(pos["sym"])
        if it:
            pos["price"] = round(float(it.marketPrice), 2)
            pos["value"] = round(float(it.marketValue), 0)
            pos["pnl"] = round(float(it.unrealizedPNL), 0)
            pos["pnl_pct"] = round((pos["price"] / pos["avg_cost"] - 1) * 100, 2) if pos["avg_cost"] else None

    ib.disconnect()

    today = datetime.now().strftime("%Y-%m-%d")
    snap = {"inception": None, "history": []}
    if os.path.exists(SNAP):
        with open(SNAP, encoding="utf-8") as f:
            snap = json.load(f)
    if not snap.get("inception"):
        snap["inception"] = {"date": today, "equity": netliq}
    snap.update({
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "account": accounts[0] if accounts else "?",
        "is_paper": bool(accounts and accounts[0].startswith(("DU", "DF"))),
        "equity": netliq, "cash": cash, "positions": positions,
    })
    hist = [h for h in snap["history"] if h["date"] != today]
    hist.append({"date": today, "equity": netliq})
    snap["history"] = hist[-500:]

    with open(SNAP, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
    print(f"נשמר: {accounts} equity=${netliq:,.0f}, {len(positions)} פוזיציות")
    _on_success()

    if not args.no_push:
        try:
            subprocess.run(["git", "pull", "--quiet"], cwd=BASE, timeout=60)
            subprocess.run(["git", "add", "ibkr_snapshot.json"], cwd=BASE, timeout=30)
            r = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=BASE)
            if r.returncode != 0:
                subprocess.run(["git", "commit", "-m", f"IBKR sync {today} [skip ci]",
                                "--quiet"], cwd=BASE, timeout=30)
                subprocess.run(["git", "push", "--quiet"], cwd=BASE, timeout=90)
                print("נדחף לריפו — הדשבורד בענן יתעדכן.")
            else:
                print("אין שינוי — לא נדחף.")
        except Exception as e:
            print(f"אזהרה: push נכשל ({e}) — הקובץ נשמר מקומית.")


if __name__ == "__main__":
    main()
