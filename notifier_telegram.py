# -*- coding: utf-8 -*-
"""
שולח איתותי מסחר יומיים לטלגרם.

קונפיגורציה (בסדר עדיפות):
  1. משתני סביבה: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  (מתאים ל-GitHub Actions / Secrets)
  2. קובץ מקומי telegram_config.json:
     { "enabled": true, "bot_token": "123:ABC...", "chat_id": "123456789" }

אם אין טוקן/chat_id — הפונקציה לא עושה כלום (no-op).
משתמש רק בספריית התקן (urllib) — אין תלות נוספת.

איך מקבלים טוקן ו-chat_id:
  1. בטלגרם, דבר עם @BotFather → /newbot → קבל BOT TOKEN.
  2. שלח הודעה כלשהי לבוט שיצרת.
  3. גש ל: https://api.telegram.org/bot<TOKEN>/getUpdates → ה-chat_id מופיע תחת result[].message.chat.id
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime

from securities import hname, dashboard_url

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "telegram_config.json")
API = "https://api.telegram.org/bot{token}/sendMessage"


def _load_config() -> dict | None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat  = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat:
        return {"bot_token": token, "chat_id": chat}

    if not os.path.exists(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"  [telegram] שגיאה בקריאת telegram_config.json: {e}")
        return None
    if not cfg.get("enabled"):
        return None
    if not cfg.get("bot_token") or not cfg.get("chat_id"):
        print("  [telegram] telegram_config.json חסר bot_token/chat_id")
        return None
    return cfg


def _fmt_pct(v: float) -> str:
    return f"{v:+.2f}%"


def _build_text(payload: dict) -> str:
    d        = payload["date"]
    regime   = payload["regime"]
    port_val = payload["portfolio_val"]
    cum_pct  = payload["cum_pct"]
    cash     = payload["cash"]
    bonds    = payload["bonds"]
    equity   = payload["equity"]
    entries  = payload["entries"]
    exits    = payload["exits"]
    holdings = payload["holdings"]
    projected = payload.get("projected")
    bond_instruments = payload.get("bond_instruments", [])
    bond_action = payload.get("bond_action")

    regime_he = {"BULL": "שורי 🐂", "NEUTRAL": "ניטרלי", "BEAR": "דובי 🐻"}.get(regime, regime)
    reason_he = {"trail_stop": "Trail Stop", "take_profit": "Take Profit",
                 "signal_exit": "יציאת איתות", "regime_bear": "שוק דובי"}

    L = []
    L.append(f'📊 <b>איתותי ת"א 125 — {d}</b>')
    L.append(f"מצב שוק: {regime_he}")
    L.append(f"שווי תיק: ₪{port_val:,.0f} | תשואה מצטברת: {_fmt_pct(cum_pct)}")
    L.append("")

    # ── קניות ──
    if entries:
        L.append(f"🟢 <b>קניות ({len(entries)})</b>")
        for e in entries:
            prices = [lp for (_t, lp, _q) in e["tranches"]]
            lo, hi = (min(prices), max(prices)) if prices else (e["base_price"], e["base_price"])
            L.append(f"• <b>{hname(e['sym'])}</b>")
            L.append(f"  {e['qty_total']:,.0f} יח׳ | טווח ₪{lo:,.0f}–₪{hi:,.0f} | {e['pos_pct']:.1f}% מהתיק")
            ladder = " · ".join(
                f"T{t}: {q:,.0f}@₪{lp:,.0f}" + ("(שוק)" if t == 1 else "")
                for (t, lp, q) in e["tranches"]
            )
            L.append(f"  <i>{ladder}</i>")
    else:
        L.append("🟢 <b>קניות</b>: אין כניסות חדשות היום.")
    L.append("")

    # ── מכירות ──
    if exits:
        L.append(f"🔴 <b>מכירות ({len(exits)})</b>")
        for e in exits:
            L.append(f"• <b>{hname(e['sym'])}</b>")
            L.append(f"  {e['qty']:,.0f} יח׳ | טווח מכירה ₪{e['sell_low']:,.0f}–₪{e['sell_high']:,.0f}"
                     f" | {reason_he.get(e['reason'], e['reason'])} | {_fmt_pct(e['pnl_pct'])}")
    else:
        L.append("🔴 <b>מכירות</b>: אין יציאות היום.")
    L.append("")

    # ── אג"ח (איזון מחדש) ──
    if bond_action:
        bonds_names = " · ".join(f'{n} (נ"ע {sn})' for n, sn in bond_instruments) or "אג\"ח ממשלתי"
        if bond_action["side"] == "BUY":
            L.append(f'🟢 <b>קניית אג"ח: ₪{bond_action["amount"]:,.0f}</b> '
                     f'(יעד {bond_action["target_pct"]:.0f}% מהתיק)')
        else:
            L.append(f'🔴 <b>מכירת אג"ח: ₪{bond_action["amount"]:,.0f}</b> '
                     f'(יעד {bond_action["target_pct"]:.0f}% מהתיק)')
        L.append(f"  <i>{bonds_names}</i>")
        L.append("")

    # ── תיק מומלץ אחרי ביצוע ──
    if projected:
        tot = projected["total"] or 1
        L.append("🎯 <b>הרכב התיק המומלץ לאחר ביצוע</b>")
        for (s, qty, val) in projected["stocks"]:
            L.append(f"• {hname(s)}: {qty:,.0f} יח׳ — ₪{val:,.0f} ({val/tot*100:.1f}%)")
        if projected["bonds"] > 0 and bond_instruments:
            split = projected["bonds"] / len(bond_instruments)
            for (name, secn) in bond_instruments:
                L.append(f'• {name} · נ"ע {secn}: ₪{split:,.0f} ({split/tot*100:.1f}%)')
        elif projected["bonds"] > 0:
            L.append(f'• אג"ח ממשלתי: ₪{projected["bonds"]:,.0f} ({projected["bonds"]/tot*100:.1f}%)')
        L.append(f"• מזומן: ₪{projected['cash']:,.0f} ({projected['cash']/tot*100:.1f}%)")
        L.append(f'סה"כ: ₪{projected["total"]:,.0f}')
        L.append("")

    url = dashboard_url()
    if url:
        L.append(f'📈 <a href="{url}">צפייה בדשבורד המלא</a>')
        L.append("")
    L.append("<i>אינה ייעוץ השקעות. הביצוע באחריותך דרך הברוקר שלך.</i>")
    return "\n".join(L)


def send_signals_telegram(payload: dict) -> bool:
    """שולח את איתותי היום לטלגרם. מחזיר True אם נשלח."""
    cfg = _load_config()
    if cfg is None:
        return False
    try:
        text = _build_text(payload)
        data = urllib.parse.urlencode({
            "chat_id": cfg["chat_id"],
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }).encode("utf-8")
        url = API.format(token=cfg["bot_token"])
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=30) as resp:
            ok = json.loads(resp.read().decode("utf-8")).get("ok", False)
        if ok:
            print("  [telegram] איתותים נשלחו בהצלחה")
        else:
            print("  [telegram] טלגרם החזיר ok=false")
        return bool(ok)
    except Exception as e:
        print(f"  [telegram] שליחה נכשלה: {e}")
        return False


if __name__ == "__main__":
    demo = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "regime": "NEUTRAL", "portfolio_val": 98443, "cum_pct": -1.56,
        "cash": 0, "bonds": 40060, "equity": 78383,
        "entries": [{"sym": "POLI.TA", "base_price": 7250, "pos_pct": 8.0, "rs": 3.2,
                     "qty_total": 11, "tranches": [(1, 7250, 6), (2, 7178, 3), (3, 7105, 2)]}],
        "exits": [{"sym": "TEVA.TA", "reason": "trail_stop", "pnl_pct": -4.1,
                   "qty": 8, "price": 1820, "sell_low": 1802, "sell_high": 1820}],
        "holdings": [("RSEL.TA", 5, 1110, 1125, 1.35)],
        "projected": {"total": 98443, "equity": 78383, "bonds": 40060, "cash": 0,
                      "stocks": [("POLI.TA", 11, 79750), ("RSEL.TA", 5, 5625)]},
        "bond_instruments": [("ממשלתי שקלי 0327 — שקלי קצר", "1139344")],
    }
    print("נשלח" if send_signals_telegram(demo) else "לא נשלח (הגדר TELEGRAM_BOT_TOKEN/CHAT_ID או telegram_config.json)")
