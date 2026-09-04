# -*- coding: utf-8 -*-
"""
בדיקת מוכנות שבועית — האם IB Gateway מחובר לפני איתות יום שני?
רץ בימי שני 14:30 (לפני האיתות של 15:07) ושולח לטלגרם סטטוס:
  מחובר  -> אישור קצר שהכול מוכן
  מנותק  -> התראה לפתוח את הגייטוואי לפני שמגיעות פקודות
"""
from __future__ import annotations
import json
import os
import sys
import urllib.parse
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
TG_CONFIG = os.path.join(BASE, "telegram_config.json")


def tg_send(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat) and os.path.exists(TG_CONFIG):
        with open(TG_CONFIG, encoding="utf-8") as f:
            c = json.load(f)
        token, chat = c.get("bot_token"), c.get("chat_id")
    if not (token and chat):
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text,
                                   "parse_mode": "HTML"}).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage", data=data)
    urllib.request.urlopen(req, timeout=30)


def main():
    connected, account = False, None
    try:
        from ib_async import IB
        ib = IB()
        for port in (4002, 7497):
            try:
                ib.connect("127.0.0.1", port, clientId=23, timeout=8)
                accounts = ib.managedAccounts()
                account = accounts[0] if accounts else "?"
                connected = True
                ib.disconnect()
                break
            except Exception:
                continue
    except Exception:
        pass

    if connected:
        tg_send(f'🟢 <b>בדיקת מוכנות שבועית</b>\n'
                f'IB Gateway מחובר ({account}) — מוכן לפקודות של היום.\n'
                f'האיתות יגיע ~15:07; אחריו הרץ את ה-executor ואשר בטלגרם.')
        print(f"מחובר ({account})")
    else:
        tg_send('🔴 <b>בדיקת מוכנות שבועית</b>\n'
                'IB Gateway לא מחובר! פתח אותו והתחבר לפני האיתות של 15:07,\n'
                'אחרת לא יהיה אפשר לשלוח את פקודות השבוע ל-IBKR.')
        print("לא מחובר — נשלחה התראה")


if __name__ == "__main__":
    main()
