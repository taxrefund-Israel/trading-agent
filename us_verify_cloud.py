# -*- coding: utf-8 -*-
"""
בדיקת תקינות שבועית בענן (GitHub Actions) — לא תלויה במחשב הביתי.
רץ בימי שני ~17:37 IL, אחרי חלון האיתות+הביצוע, ובודק מתוך קבצי הריפו:
  1. האם איתות היום רץ (שורת history ב-us_portfolio_state.json)
  2. האם היו עסקאות היום, ואם כן — האם ה-snapshot של IBKR עודכן אחרי הביצוע
  3. השוואת פוזיציות IBKR מול יעד הסוכן (בקנה המידה של החשבון)
שולח סיכום לטלגרם תמיד — גם כשהכול תקין וגם בשבוע שקט.

הרצה מקומית לבדיקה: python us_verify_cloud.py --no-send
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(BASE, "us_portfolio_state.json")
SNAP = os.path.join(BASE, "ibkr_snapshot.json")
TG_CONFIG = os.path.join(BASE, "telegram_config.json")
TOLERANCE = 0.07   # פער כמות מותר מול היעד (7%)


def tg_send(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat) and os.path.exists(TG_CONFIG):
        with open(TG_CONFIG, encoding="utf-8") as f:
            c = json.load(f)
        token, chat = c.get("bot_token"), c.get("chat_id")
    if not (token and chat):
        return False
    data = urllib.parse.urlencode({"chat_id": chat, "text": text,
                                   "parse_mode": "HTML"}).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage", data=data)
    with urllib.request.urlopen(req, timeout=30) as r:
        return bool(json.loads(r.read().decode()).get("ok"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-send", action="store_true")
    args = ap.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    issues, infos = [], []

    state = snap = None
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            state = json.load(f)
    if os.path.exists(SNAP):
        with open(SNAP, encoding="utf-8") as f:
            snap = json.load(f)

    # 1. איתות היום
    signal_ran = bool(state and any(h["date"] == today for h in state.get("history", [])))
    if signal_ran:
        row = next(h for h in state["history"] if h["date"] == today)
        infos.append(f'איתות הענן רץ ✓ (משטר {"שורי" if row.get("bull") else "דובי"}, '
                     f'אירוע: {row.get("event", "?")})')
    else:
        issues.append("איתות הענן של היום לא נמצא בריפו — בדוק את us-weekly-signals ב-Actions")

    # 2. עסקאות היום + טריות ה-snapshot
    trades_today = [t for t in (state or {}).get("trades", []) if t["date"] == today]
    snap_date = (snap or {}).get("updated", "")[:10]
    snap_fresh = snap_date == today

    if trades_today:
        n_b = sum(1 for t in trades_today if t["side"] == "BUY")
        n_s = len(trades_today) - n_b
        infos.append(f"עסקאות היום: {n_b} קניות, {n_s} מכירות")
        if snap_fresh:
            infos.append("snapshot של IBKR עודכן אחרי הביצוע ✓")
        else:
            issues.append("יש עסקאות היום אבל ה-snapshot של IBKR לא עודכן — "
                          "ייתכן שהביצוע לא אושר/לא רץ (מחשב כבוי? Gateway סגור?)")
    else:
        infos.append("שבוע שקט — אין עסקאות היום, התיק ללא שינוי")

    # 3. פוזיציות מול יעד (על ה-snapshot האחרון הקיים)
    if state and snap and state.get("positions") and snap.get("equity"):
        agent_val = state["history"][-1]["value"] if state.get("history") else None
        if agent_val:
            scale = snap["equity"] / agent_val
            ib_pos = {p["sym"]: p["qty"] for p in snap.get("positions", [])}
            mism = []
            for sym, h in state["positions"].items():
                target = h["qty"] * scale
                actual = ib_pos.get(sym, 0)
                if target > 0 and abs(actual - target) / target > TOLERANCE:
                    mism.append(f"{sym}: יעד ~{target:,.0f} מול {actual:,.0f} בפועל")
            extra = [s for s in ib_pos if s not in state["positions"]]
            if extra:
                mism.append(f"ניירות עודפים ב-IBKR: {', '.join(extra)}")
            if mism:
                issues.append("פערי פוזיציות מול היעד: " + " | ".join(mism))
            else:
                infos.append(f'פוזיציות IBKR תואמות ליעד ✓ '
                             f'({len(state["positions"])} ניירות, מקדם x{scale:.1f}, '
                             f'נכון ל-{snap_date})')

    ok = not issues
    L = [f'🔍 <b>בדיקת תקינות שבועית (ענן) — {today}</b>', ""]
    if ok:
        L.append("🟢 הכול רץ כמו שצריך, אין פעולות נדרשות.")
    else:
        L.append("🔴 נמצאו בעיות:")
        L += [f"• {i}" for i in issues]
    L.append("")
    L += [f"· {i}" for i in infos]
    if snap and snap.get("equity"):
        L.append("")
        L.append(f'שווי חשבון IBKR: ${snap["equity"]:,.0f} (נכון ל-{snap_date})')
    msg = "\n".join(L)

    print(msg)
    if not args.no_send:
        sent = tg_send(msg)
        print(f"\n[telegram] {'נשלח' if sent else 'לא נשלח'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
