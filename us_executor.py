# -*- coding: utf-8 -*-
"""
US Executor — רמה 2: ביצוע חצי-אוטומטי של החלטות הסוכן ב-Interactive Brokers,
עם אישור בטלגרם לפני כל שליחה.

זרימה:
  1. מושך את הריפו (החלטות הריצה האחרונה של הסוכן בענן).
  2. קורא את עסקאות היום מ-us_portfolio_state.json — אלה ההמלצות.
  3. בדיקות שפיות: חיבור ל-IB, חשבון דמו בלבד (אלא אם הותר אחרת),
     סטיית מחיר מול מחיר ההחלטה.
  4. שולח לטלגרם את רשימת הפקודות עם כפתורי אישור/ביטול.
  5. רק אחרי לחיצה על ✅ — שולח פקודות Limit ל-IBKR ומדווח על הביצועים.

ברירת מחדל: חשבון PAPER בלבד (פורט 7497). חשבון אמיתי חסום אלא אם
"allow_live": true בקובץ הקונפיג — אל תדליק בלי לחשוב פעמיים.

הרצה: python us_executor.py            (עסקאות של היום)
       python us_executor.py --date 2026-09-05
       python us_executor.py --dry-run  (הכול חוץ משליחה בפועל ל-IB)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE        = os.path.dirname(os.path.abspath(__file__))
STATE_FILE  = os.path.join(BASE, "us_portfolio_state.json")
CFG_FILE    = os.path.join(BASE, "us_executor_config.json")
TG_CONFIG   = os.path.join(BASE, "telegram_config.json")

DEFAULT_CFG = {
    "host": "127.0.0.1",
    "port": 7497,              # 7497 = TWS paper, 4002 = Gateway paper
    "client_id": 21,
    "allow_live": False,       # לעולם לא true בלי החלטה מפורשת
    "max_price_dev_pct": 3.0,  # סטייה מקסימלית ממחיר ההחלטה
    "limit_buffer_pct": 0.3,   # מרווח ה-Limit מעל/מתחת למחיר הנוכחי
    "approval_timeout_min": 30,
}


# ─── קונפיג + טלגרם ──────────────────────────────────────────────────────────
def load_cfg() -> dict:
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(CFG_FILE):
        with open(CFG_FILE, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def tg_creds():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat) and os.path.exists(TG_CONFIG):
        with open(TG_CONFIG, encoding="utf-8") as f:
            c = json.load(f)
        token, chat = c.get("bot_token"), c.get("chat_id")
    if not (token and chat):
        raise RuntimeError("אין הגדרות טלגרם")
    return token, str(chat)


def tg_api(token, method, payload):
    data = urllib.parse.urlencode(payload).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/{method}", data=data)
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode())


def tg_send(token, chat, text, keyboard=None):
    p = {"chat_id": chat, "text": text, "parse_mode": "HTML"}
    if keyboard:
        p["reply_markup"] = json.dumps(keyboard)
    return tg_api(token, "sendMessage", p)


def wait_for_approval(token, chat, run_id, timeout_min) -> bool:
    """ממתין ללחיצת כפתור בטלגרם. True=אושר, False=בוטל/פג תוקף."""
    deadline = time.time() + timeout_min * 60
    offset = None
    while time.time() < deadline:
        p = {"timeout": 30, "allowed_updates": json.dumps(["callback_query"])}
        if offset:
            p["offset"] = offset
        try:
            resp = tg_api(token, "getUpdates", p)
        except Exception:
            time.sleep(5)
            continue
        for upd in resp.get("result", []):
            offset = upd["update_id"] + 1
            cq = upd.get("callback_query")
            if not cq:
                continue
            if str(cq.get("from", {}).get("id")) != chat and \
               str(cq.get("message", {}).get("chat", {}).get("id")) != chat:
                continue  # לא מהצ'אט המורשה
            data = cq.get("data", "")
            if data == f"exec_ok_{run_id}":
                tg_api(token, "answerCallbackQuery",
                       {"callback_query_id": cq["id"], "text": "אושר — שולח ל-IBKR"})
                return True
            if data == f"exec_no_{run_id}":
                tg_api(token, "answerCallbackQuery",
                       {"callback_query_id": cq["id"], "text": "בוטל"})
                return False
    return False


# ─── החלטות לביצוע ───────────────────────────────────────────────────────────
def load_agent_state():
    with open(STATE_FILE, encoding="utf-8") as f:
        state = json.load(f)
    agent_val = state["history"][-1]["value"] if state.get("history") else 100_000.0
    return state, agent_val


def todays_orders(date_str) -> list[dict]:
    state, _ = load_agent_state()
    return [t for t in state["trades"] if t["date"] == date_str]


def ib_symbol(sym: str) -> str:
    return sym.replace("-", " ")   # BRK-B -> BRK B


def build_target_orders(ib, cfg, log) -> tuple[list[dict], float, float]:
    """מצב sync-target: משווה את פוזיציות IBKR לתיק הסוכן, מוקטן/מוגדל
    לפי ההון בחשבון (NetLiquidation), ומחזיר פקודות דלתא."""
    from ib_async import Stock
    from backtest_us_v1 import UNIVERSE

    state, agent_val = load_agent_state()
    netliq = None
    for v in ib.accountValues():
        if v.tag == "NetLiquidation" and v.currency == "USD":
            netliq = float(v.value)
    if not netliq or netliq <= 0:
        raise RuntimeError("לא הצלחתי לקרוא NetLiquidation מהחשבון")
    scale = netliq / agent_val
    log(f"הון בחשבון: ${netliq:,.0f} | תיק הסוכן: ${agent_val:,.0f} | מקדם: x{scale:.2f}")

    ib_pos = {}
    for p in ib.positions():
        ib_pos[p.contract.symbol] = ib_pos.get(p.contract.symbol, 0) + int(p.position)

    universe_ib = {ib_symbol(s): s for s in UNIVERSE}
    orders = []
    # יעדים לפי החזקות הסוכן
    for sym, h in state["positions"].items():
        target = int(h["qty"] * scale)
        cur = ib_pos.get(ib_symbol(sym), 0)
        delta = target - cur
        if delta == 0:
            continue
        c = Stock(ib_symbol(sym), "SMART", "USD")
        ib.qualifyContracts(c)
        price = live_price(ib, c)
        if not price:
            log(f"  {sym}: אין מחיר — מדולג")
            continue
        orders.append({"side": "BUY" if delta > 0 else "SELL",
                       "sym": sym, "qty": abs(delta), "price": price})
    # ניירות מהיקום שקיימים ב-IB אך לא בתיק הסוכן — למכור
    agent_ib_syms = {ib_symbol(s) for s in state["positions"]}
    for ibsym, qty in ib_pos.items():
        if ibsym in universe_ib and ibsym not in agent_ib_syms and qty > 0:
            c = Stock(ibsym, "SMART", "USD")
            ib.qualifyContracts(c)
            price = live_price(ib, c)
            if price:
                orders.append({"side": "SELL", "sym": universe_ib[ibsym],
                               "qty": qty, "price": price})
    return orders, netliq, scale


# ─── IB ──────────────────────────────────────────────────────────────────────
def connect_ib(cfg):
    from ib_async import IB
    ib = IB()
    ports = [cfg["port"]] + [p for p in (4002, 7497) if p != cfg["port"]]
    last_err = None
    for port in ports:
        try:
            ib.connect(cfg["host"], port, clientId=cfg["client_id"], timeout=15)
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"אין חיבור ל-IB באף פורט {ports}: {last_err}")
    accounts = ib.managedAccounts()
    is_paper = all(a.startswith(("DU", "DF")) for a in accounts)
    if not is_paper and not cfg.get("allow_live"):
        ib.disconnect()
        raise RuntimeError(
            f"חשבון {accounts} אינו חשבון דמו, ו-allow_live כבוי. מסרב לבצע.")
    return ib, accounts, is_paper


def live_price(ib, contract):
    """מחיר עדכני (גם delayed בסדר לחשבון דמו)."""
    ib.reqMarketDataType(3)  # 3 = delayed אם אין מנוי real-time
    t = ib.reqMktData(contract, "", False, False)
    ib.sleep(2.5)
    for attr in ("last", "close", "bid"):
        v = getattr(t, attr, None)
        if v and v == v and v > 0:
            return float(v)
    return None


def place_orders(ib, orders, cfg, log):
    from ib_async import Stock, LimitOrder
    results = []
    for o in orders:
        sym = o["sym"].replace("-", " ")  # BRK-B -> BRK B בסימון IB
        contract = Stock(sym, "SMART", "USD")
        ib.qualifyContracts(contract)
        cur = live_price(ib, contract)
        dev = abs(cur / o["price"] - 1) * 100 if cur else None
        if cur is None:
            results.append((o, None, "אין מחיר — דולג"))
            continue
        if dev > cfg["max_price_dev_pct"]:
            results.append((o, None, f"סטייה {dev:.1f}% ממחיר ההחלטה — דולג"))
            continue
        buf = cfg["limit_buffer_pct"] / 100
        if o["side"] == "BUY":
            lmt = round(cur * (1 + buf), 2)
            order = LimitOrder("BUY", o["qty"], lmt)
        else:
            lmt = round(cur * (1 - buf), 2)
            order = LimitOrder("SELL", o["qty"], lmt)
        order.tif = "DAY"
        trade = ib.placeOrder(contract, order)
        ib.sleep(1.5)
        st = trade.orderStatus.status
        results.append((o, lmt, st))
        log(f"  {o['side']} {o['sym']} x{o['qty']} limit {lmt} -> {st}")
    return results


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sync-target", action="store_true",
                    help="יישור מלא של חשבון IBKR לתיק הסוכן, מוקנה להון בחשבון")
    ap.add_argument("--wait-signal", type=int, default=0, metavar="MIN",
                    help="המתן עד N דקות לאיתות הענן של היום (משיכת git חוזרת)")
    args = ap.parse_args()

    def log(m):
        print(m)

    cfg = load_cfg()
    token, chat = tg_creds()

    # מנעול כפילות: אם כבר בוצעו פקודות היום — לא רצים שוב
    marker = os.path.join(BASE, "logs_us", f"executed_{args.date}.marker")
    if os.path.exists(marker) and not args.dry_run:
        log(f"כבר בוצעו פקודות בתאריך {args.date} (קיים marker) — יציאה.")
        return

    # 1. סנכרון עם החלטות הענן (+ המתנה לאיתות אם התבקשה)
    def pull():
        try:
            subprocess.run(["git", "pull", "--quiet"], cwd=BASE, timeout=60)
        except Exception as e:
            log(f"אזהרה: git pull נכשל ({e}) — ממשיך עם הסטייט המקומי")

    pull()
    if args.wait_signal > 0:
        deadline = time.time() + args.wait_signal * 60
        while time.time() < deadline:
            state, _ = load_agent_state()
            if any(h["date"] == args.date for h in state.get("history", [])):
                log("איתות היום התקבל בריפו — ממשיך.")
                break
            log("אין עדיין איתות להיום — ממתין 2 דקות...")
            time.sleep(120)
            pull()
        else:
            log(f"לא הגיע איתות תוך {args.wait_signal} דק' — יציאה.")
            tg_send(token, chat,
                    f'⚠️ <b>Executor</b> — האיתות של {args.date} לא הגיע מהענן '
                    f'תוך {args.wait_signal} דקות. בדוק את ריצת GitHub Actions.')
            return

    # 2. חיבור ל-IB ובדיקת חשבון
    if args.dry_run:
        ib, accounts, is_paper = None, ["DRY-RUN"], True
    else:
        ib, accounts, is_paper = connect_ib(cfg)
        log(f"מחובר ל-IB: {accounts} ({'דמו' if is_paper else 'אמיתי!'})")

    # 3. בניית הפקודות
    sync_note = ""
    if args.sync_target:
        if ib is None:
            raise RuntimeError("--sync-target דורש חיבור אמיתי ל-IB (לא dry-run)")
        orders, netliq, scale = build_target_orders(ib, cfg, log)
        sync_note = f'יישור תיק להון בחשבון: ${netliq:,.0f} (מקדם x{scale:.2f})'
        if not orders:
            tg_send(token, chat, "✅ חשבון IBKR כבר מיושר לתיק הסוכן — אין פקודות.")
            log("מיושר — אין מה לבצע.")
            ib.disconnect()
            return
    else:
        orders = todays_orders(args.date)
        if not orders:
            log(f"אין עסקאות בתאריך {args.date} — אין מה לבצע.")
            tg_send(token, chat,
                    f'✋ <b>Executor — {args.date}</b>\n'
                    f'אין פקודות לביצוע השבוע. התיק ב-IBKR נשאר ללא שינוי.')
            if ib:
                ib.disconnect()
            return

    # 4. בקשת אישור בטלגרם
    run_id = args.date.replace("-", "") + ("s" if args.sync_target else "")
    acct_tag = "🧪 חשבון דמו" if is_paper else "⚠️ חשבון אמיתי"
    L = [f'🔐 <b>אישור ביצוע ב-IBKR — {args.date}</b>', acct_tag]
    if sync_note:
        L.append(sync_note)
    L.append("")
    for o in orders:
        emoji = "🟢" if o["side"] == "BUY" else "🔴"
        L.append(f'{emoji} {o["side"]} <b>{o["sym"]}</b> — {o["qty"]} יח׳ '
                 f'@ ~${o["price"]:,.2f} = <b>${o["qty"] * o["price"]:,.0f}</b>')
    total_buy = sum(o["qty"] * o["price"] for o in orders if o["side"] == "BUY")
    total_sell = sum(o["qty"] * o["price"] for o in orders if o["side"] == "SELL")
    L.append("")
    if total_buy:
        L.append(f'סה"כ רכישות בסבב: <b>${total_buy:,.0f}</b>')
    if total_sell:
        L.append(f'סה"כ מימושים בסבב: <b>${total_sell:,.0f}</b>')
    L += [f'פקודות Limit עם מרווח {cfg["limit_buffer_pct"]}%, '
              f'ביטול אוטומטי אם סטייה מעל {cfg["max_price_dev_pct"]}%.',
          f'תוקף האישור: {cfg["approval_timeout_min"]} דקות.']
    kb = {"inline_keyboard": [[
        {"text": "✅ אשר ושלח ל-IBKR", "callback_data": f"exec_ok_{run_id}"},
        {"text": "❌ בטל", "callback_data": f"exec_no_{run_id}"},
    ]]}
    tg_send(token, chat, "\n".join(L), kb)
    log("נשלחה בקשת אישור לטלגרם. ממתין...")

    approved = wait_for_approval(token, chat, run_id, cfg["approval_timeout_min"])
    if not approved:
        tg_send(token, chat, "❌ הביצוע בוטל / פג תוקף האישור. לא נשלחו פקודות.")
        log("לא אושר — יציאה.")
        if ib:
            ib.disconnect()
        return

    if args.dry_run:
        tg_send(token, chat, "🧪 dry-run: אושר, אך לא נשלחו פקודות (מצב בדיקה).")
        log("dry-run — סיום.")
        return

    # 4. ביצוע ודיווח
    results = place_orders(ib, orders, cfg, log)
    ib.sleep(3)
    R = [f'📬 <b>דו"ח ביצוע IBKR — {args.date}</b>']
    for o, lmt, st in results:
        if lmt is None:
            R.append(f'⚠️ {o["side"]} {o["sym"]}: {st}')
        else:
            R.append(f'{"🟢" if o["side"]=="BUY" else "🔴"} {o["side"]} {o["sym"]} '
                     f'x{o["qty"]} limit ${lmt} — {st}')
    R.append("\nבדוק ב-TWS שהפקודות מולאו. פקודות DAY שלא ימולאו יפוגו בסוף היום.")
    tg_send(token, chat, "\n".join(R))
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w") as f:
        f.write(datetime.now().isoformat())
    ib.disconnect()
    # דחיפת snapshot טרי לריפו — כדי שהמוודא בענן יראה את הביצוע
    try:
        subprocess.run([sys.executable, os.path.join(BASE, "us_ibkr_sync.py")],
                       cwd=BASE, timeout=180)
    except Exception as e:
        log(f"אזהרה: סנכרון snapshot אחרי ביצוע נכשל ({e})")
    log("סיום.")


if __name__ == "__main__":
    main()
