# -*- coding: utf-8 -*-
"""
US Momentum Agent — הפעלה חיה של אסטרטגיית המומנטום האמריקאית.

אסטרטגיה (מאומתת ב-backtest_us_v1/v2/v3/v4):
  * בדיקת משטר שבועית: S&P 500 מעל/מתחת SMA200 → מניות / קרן כספית
  * ריבאלנס מומנטום חודשי (בריצה הראשונה של כל חודש קלנדרי):
    מחזיקים Top-5 לפי מומנטום 12-1, מוכרים רק כשמניה נופלת מתחת לדירוג 12
  * חזרה מיידית לשוק בהתאוששות (גם באמצע חודש)

מנהל תיק נייר ב-us_portfolio_state.json ושולח התראת טלגרם שבועית.
מיועד לריצה שבועית (יום ב' 15:00 IL, לפני פתיחת וול-סטריט, על נתוני סגירת שישי).
"""
from __future__ import annotations

import json
import os
import socket
import sys
import urllib.parse
import urllib.request
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_us_v1 import UNIVERSE       # single source of truth for the universe
from backtest_us_v6_next import SECTOR    # sector map for the diversification cap

# ─── קונפיגורציה (v6, מ-2026-09-05): Top6 + תקרת סקטור 1 ─────────────────────
TOP_N        = 6
KEEP_RANK    = 14
SECTOR_CAP   = 1   # מקסימום מניה אחת מכל סקטור — הלקח מקריסת המוליכים-למחצה
COMMISSION   = 0.0005
TAX_RATE     = 0.25
INITIAL_CASH = 100_000.0

BASE       = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE, "us_portfolio_state.json")
LOG_DIR    = os.path.join(BASE, "logs_us")
TG_CONFIG  = os.path.join(BASE, "telegram_config.json")

# קרן כספית דולרית ספציפית לחניית מזומן (הזולה בקטגוריה הנקובה, 07/2026)
MMF_REC = ('איילון כספית דולרית נקובה · מס׳ קרן 5139076 · ד"נ 0.13%\n'
           '  (חלופה בברוקר אמריקאי: SGOV — iShares 0-3 Month Treasury ETF)')


def lan_ip() -> str:
    """כתובת ה-IP ברשת הביתית — לקישורי דשבורד שנפתחים מהטלפון."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


# ─── נתונים ───────────────────────────────────────────────────────────────────
def fetch_prices() -> pd.DataFrame:
    """מחירי סגירה מתואמים ~420 ימי מסחר אחורה לכל היקום + מדדים."""
    tickers = UNIVERSE + ["^GSPC", "^DJI", "^NDX"]
    df = yf.download(tickers, period="2y", interval="1d",
                     auto_adjust=True, progress=False)["Close"]
    df.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in df.index])
    return df.ffill()


def momentum_ranks(px: pd.DataFrame) -> pd.Series:
    """מומנטום 12-1: תשואה מ-t-252 עד t-21. ממוין יורד."""
    stocks = [c for c in px.columns if not c.startswith("^")]
    if len(px) < 253:
        raise RuntimeError("אין מספיק היסטוריה למומנטום 12-1")
    mom = (px[stocks].iloc[-21] / px[stocks].iloc[-252] - 1).dropna()
    return mom.sort_values(ascending=False)


def regime_status(px: pd.DataFrame) -> dict:
    """משטר היברידי (v5b): שורי רק כששני המדדים — SPX וגם NDX — מעל SMA200.
    יציאה דובית כשאחד מהם שובר. תשואה כמו משטר NDX עם ה-Drawdown הנמוך ביותר."""
    out = {}
    for key, tk in (("spx", "^GSPC"), ("ndx", "^NDX")):
        s = px[tk].dropna()
        sma = float(s.rolling(200).mean().iloc[-1])
        c = float(s.iloc[-1])
        out[f"{key}_dist"] = (c / sma - 1) * 100
        out[f"{key}_bull"] = c > sma
    out["bull"] = out["spx_bull"] and out["ndx_bull"]
    return out


# ─── סטייט ────────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"cash": INITIAL_CASH, "positions": {}, "trades": [],
            "history": [], "bull": None, "last_rebal_month": None,
            "inception": None, "tax_paid": 0.0}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ─── פעולות תיק ───────────────────────────────────────────────────────────────
def do_sell(state, sym, price, reason, today):
    h = state["positions"].pop(sym)
    proceeds = h["qty"] * price
    sc = proceeds * COMMISSION
    gain = proceeds - h["cost"] - h["comm"] - sc
    tax = max(0.0, gain * TAX_RATE)
    state["cash"] += proceeds - sc - tax
    state["tax_paid"] += tax
    pnl_pct = (price / (h["cost"] / h["qty"]) - 1) * 100
    t = {"date": today, "side": "SELL", "sym": sym, "qty": h["qty"],
         "price": round(price, 2), "tax": round(tax, 2),
         "pnl_pct": round(pnl_pct, 2), "reason": reason}
    state["trades"].append(t)
    return t


def do_buys(state, ranked, prices, today, reason):
    """ממלא סלוטים פנויים מהדירוג, חלוקה שווה של המזומן."""
    actions = []
    slots = TOP_N - len(state["positions"])
    if slots <= 0:
        return actions
    pool = [s for s in ranked.index
            if s not in state["positions"] and pd.notna(prices.get(s))]
    if not pool:
        return actions
    sec_count: dict[str, int] = {}
    for held in state["positions"]:
        sec = SECTOR.get(held, "?")
        sec_count[sec] = sec_count.get(sec, 0) + 1
    per_slot = state["cash"] * 0.98 / slots
    filled = 0
    for s in pool:
        if filled >= slots:
            break
        sec = SECTOR.get(s, "?")
        if sec_count.get(sec, 0) >= SECTOR_CAP:
            continue  # הסקטור כבר מיוצג — עוברים למדורגת הבאה
        p = float(prices[s])
        qty = int(per_slot / (p * (1 + COMMISSION)))
        if qty < 1:
            continue  # יקרה מדי לסלוט — עוברים למדורגת הבאה
        bc = qty * p * COMMISSION
        state["cash"] -= qty * p + bc
        state["positions"][s] = {"qty": qty, "cost": qty * p, "comm": bc,
                                 "buy_date": today, "buy_price": round(p, 2)}
        t = {"date": today, "side": "BUY", "sym": s, "qty": qty,
             "price": round(p, 2), "rank": int(ranked.index.get_loc(s)) + 1,
             "reason": reason}
        state["trades"].append(t)
        actions.append(t)
        filled += 1
        sec_count[sec] = sec_count.get(sec, 0) + 1
    return actions


def portfolio_value(state, prices) -> float:
    held = sum(h["qty"] * float(prices[s]) for s, h in state["positions"].items()
               if pd.notna(prices.get(s)))
    return state["cash"] + held


# ─── לוגיקה שבועית ────────────────────────────────────────────────────────────
def run_weekly(px: pd.DataFrame) -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    month_key = today[:7]
    state = load_state()
    prices = px.iloc[-1]
    rg = regime_status(px)
    bull = rg["bull"]
    ranked = momentum_ranks(px)
    rank_of = {s: r + 1 for r, s in enumerate(ranked.index)}

    sells, buys = [], []
    event = "status"

    prev_bull = state["bull"]
    if prev_bull is None:
        prev_bull = bull  # ריצה ראשונה — אין מעבר משטר

    # 1. מעבר משטר
    if prev_bull and not bull:
        event = "bear_exit"
        for sym in list(state["positions"].keys()):
            if pd.notna(prices.get(sym)):
                sells.append(do_sell(state, sym, float(prices[sym]),
                                     "יציאה — שוק דובי (SPX < SMA200)", today))
    elif not prev_bull and bull:
        event = "bull_entry"
        buys = do_buys(state, ranked, prices, today, "חזרה לשוק — התאוששות משטר")
        state["last_rebal_month"] = month_key

    # 2. ריבאלנס חודשי (ריצה ראשונה בחודש, רק בשוק שורי)
    if bull and event == "status" and state["last_rebal_month"] != month_key:
        event = "monthly_rebalance"
        for sym in list(state["positions"].keys()):
            r = rank_of.get(sym)
            if r is not None and r <= KEEP_RANK:
                continue
            if pd.notna(prices.get(sym)):
                sells.append(do_sell(state, sym, float(prices[sym]),
                                     f"נפלה לדירוג {r or '—'} (מעל {KEEP_RANK})", today))
        buys = do_buys(state, ranked, prices, today, "ריבאלנס חודשי")
        state["last_rebal_month"] = month_key
    elif bull and event == "bull_entry":
        state["last_rebal_month"] = month_key

    state["bull"] = bull

    # 3. עדכון היסטוריה + בנצ'מרקים
    pv = portfolio_value(state, prices)
    spx_last = float(px["^GSPC"].dropna().iloc[-1])
    dji_last = float(px["^DJI"].dropna().iloc[-1])
    ndx_last = float(px["^NDX"].dropna().iloc[-1])
    if state["inception"] is None:
        state["inception"] = {"date": today, "value": pv,
                              "spx": spx_last, "dji": dji_last, "ndx": ndx_last}
    inc = state["inception"]
    if "ndx" not in inc:  # תיק שאותחל לפני הוספת מדד הנאסד"ק
        inc["ndx"] = ndx_last
    state["history"].append({"date": today, "value": round(pv, 2),
                             "spx": spx_last, "dji": dji_last, "ndx": ndx_last,
                             "bull": bull, "event": event})
    save_state(state)

    return {
        "today": today, "event": event, "bull": bull, "regime": rg,
        "pv": pv, "cash": state["cash"],
        "cum_pct": (pv / inc["value"] - 1) * 100,
        "spx_pct": (spx_last / inc["spx"] - 1) * 100,
        "dji_pct": (dji_last / inc["dji"] - 1) * 100,
        "ndx_pct": (ndx_last / inc["ndx"] - 1) * 100,
        "sells": sells, "buys": buys,
        "positions": state["positions"], "prices": prices,
        "rank_of": rank_of, "ranked": ranked,
        "tax_paid": state["tax_paid"],
    }


# ─── טלגרם ────────────────────────────────────────────────────────────────────
def build_message(r: dict) -> str:
    event_he = {
        "status":            "אין פעולות השבוע — סטטוס בלבד",
        "monthly_rebalance": "ריבאלנס חודשי 🗓️",
        "bear_exit":         "יציאה לקרן כספית — שוק דובי 🐻",
        "bull_entry":        "חזרה לשוק — התאוששות 🐂",
    }
    rg = r["regime"]
    detail = (f'SPX {rg["spx_dist"]:+.1f}% · NDX {rg["ndx_dist"]:+.1f}% מול SMA200')
    if r["bull"]:
        regime_he = f"שורי 🐂 ({detail})"
    else:
        broke = " וגם ".join(n for n, b in (("SPX", rg["spx_bull"]), ("NDX", rg["ndx_bull"])) if not b)
        regime_he = f"דובי 🐻 — {broke} מתחת ל-SMA200 ({detail})"

    L = []
    L.append(f'🇺🇸 <b>מומנטום ארה"ב — {r["today"]}</b>')
    L.append(f'מצב שוק: {regime_he}')
    L.append(f'{event_he[r["event"]]}')
    L.append(f'שווי תיק: ${r["pv"]:,.0f} | מצטבר: {r["cum_pct"]:+.2f}% '
             f'(SPX {r["spx_pct"]:+.2f}% | נאסד"ק100 {r["ndx_pct"]:+.2f}% | דאו {r["dji_pct"]:+.2f}%)')
    L.append("")

    if r["sells"]:
        total_sell = sum(t["qty"] * t["price"] for t in r["sells"])
        L.append(f'🔴 <b>מכירות ({len(r["sells"])})</b>')
        for t in r["sells"]:
            L.append(f'• <b>{t["sym"]}</b>: {t["qty"]} יח׳ @ ${t["price"]:,.2f} '
                     f'= <b>${t["qty"] * t["price"]:,.0f}</b> '
                     f'| {t["pnl_pct"]:+.1f}% | {t["reason"]}')
        L.append(f'סה"כ מימושים: <b>${total_sell:,.0f}</b>')
        L.append("")
    if r["buys"]:
        total_buy = sum(t["qty"] * t["price"] for t in r["buys"])
        L.append(f'🟢 <b>קניות ({len(r["buys"])})</b>')
        for t in r["buys"]:
            L.append(f'• <b>{t["sym"]}</b> (דירוג #{t.get("rank", "?")}): '
                     f'{t["qty"]} יח׳ @ ${t["price"]:,.2f} '
                     f'= <b>${t["qty"] * t["price"]:,.0f}</b>')
        L.append(f'סה"כ רכישות: <b>${total_buy:,.0f}</b>')
        L.append("")
    if not r["sells"] and not r["buys"]:
        L.append("✋ אין קניות/מכירות. ממשיכים להחזיק.")
        L.append("")

    if r["positions"]:
        L.append("🎯 <b>התיק הנוכחי</b>")
        for s, h in sorted(r["positions"].items(),
                           key=lambda kv: rank_sort(kv[0], r["rank_of"])):
            p = r["prices"].get(s)
            if pd.isna(p):
                continue
            mv = h["qty"] * float(p)
            pnl = (float(p) / (h["cost"] / h["qty"]) - 1) * 100
            rk = r["rank_of"].get(s, "—")
            L.append(f'• {s}: ${mv:,.0f} | {pnl:+.1f}% | דירוג #{rk}')
    L.append(f'💵 מזומן/כספית: ${r["cash"]:,.0f}')
    # המלצת קרן כספית ספציפית — ביציאה דובית או כשיש מזומן משמעותי בצד
    if r["event"] == "bear_exit" or r["cash"] > r["pv"] * 0.10:
        L.append(f'🏦 לחניית המזומן: {MMF_REC}')
    L.append("")
    top6 = " · ".join(f'{s}#{i+1}' for i, s in enumerate(r["ranked"].index[:6]))
    L.append(f'📈 טופ-6 מומנטום: {top6}')
    L.append("")
    # דשבורד אחד מכיל את שני התיקים (ישראל למעלה, ארה"ב מתחת).
    try:
        from securities import dashboard_url
        url = dashboard_url()
    except Exception:
        url = (os.environ.get("DASHBOARD_URL") or "").strip() or None
    if url:
        L.append(f'📊 <a href="{url}">צפייה בדשבורד (ישראל + ארה"ב)</a>')
        L.append("")
    L.append("<i>תיק נייר. אינה ייעוץ השקעות — הביצוע באחריותך.</i>")
    return "\n".join(L)


def rank_sort(sym, rank_of):
    return rank_of.get(sym, 999)


def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        if not os.path.exists(TG_CONFIG):
            return False
        with open(TG_CONFIG, encoding="utf-8") as f:
            cfg = json.load(f)
        if not cfg.get("enabled"):
            return False
        token, chat = cfg.get("bot_token"), cfg.get("chat_id")
    if not (token and chat):
        return False
    data = urllib.parse.urlencode({
        "chat_id": chat, "text": text, "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode("utf-8")
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage",
                                 data=data)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return bool(json.loads(resp.read().decode()).get("ok"))
    except urllib.error.HTTPError as e:
        print(f"[telegram] שליחה נכשלה: {e} | {e.read().decode('utf-8', 'replace')}")
        return False
    except Exception as e:
        print(f"[telegram] שליחה נכשלה: {e}")
        return False


def status_snapshot(px: pd.DataFrame) -> dict:
    """בונה דוח סטטוס מהסטייט הקיים בלי לבצע עסקאות ובלי לשנות את הקובץ."""
    state = load_state()
    prices = px.iloc[-1]
    rg = regime_status(px)
    ranked = momentum_ranks(px)
    rank_of = {s: r + 1 for r, s in enumerate(ranked.index)}
    pv = portfolio_value(state, prices)
    spx_last = float(px["^GSPC"].dropna().iloc[-1])
    dji_last = float(px["^DJI"].dropna().iloc[-1])
    ndx_last = float(px["^NDX"].dropna().iloc[-1])
    inc = state["inception"] or {"value": pv, "spx": spx_last, "dji": dji_last}
    return {
        "today": datetime.now().strftime("%Y-%m-%d"), "event": "status",
        "bull": rg["bull"], "regime": rg,
        "pv": pv, "cash": state["cash"],
        "cum_pct": (pv / inc["value"] - 1) * 100,
        "spx_pct": (spx_last / inc["spx"] - 1) * 100,
        "dji_pct": (dji_last / inc["dji"] - 1) * 100,
        "ndx_pct": (ndx_last / inc.get("ndx", ndx_last) - 1) * 100,
        "sells": [], "buys": [],
        "positions": state["positions"], "prices": prices,
        "rank_of": rank_of, "ranked": ranked,
        "tax_paid": state.get("tax_paid", 0.0),
    }


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, datetime.now().strftime("%Y-%m-%d") + ".txt")

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    resend = "--resend" in sys.argv
    log(f"=== US Momentum Agent — {datetime.now():%Y-%m-%d %H:%M}"
        f"{' (resend)' if resend else ''} ===")
    try:
        px = fetch_prices()
        log(f"נתונים: {px.shape[1]} טיקרים, עד {px.index[-1]:%Y-%m-%d}")
        r = status_snapshot(px) if resend else run_weekly(px)
        log(f"אירוע: {r['event']} | משטר: {'BULL' if r['bull'] else 'BEAR'} | "
            f"תיק: ${r['pv']:,.0f} ({r['cum_pct']:+.2f}%)")
        for t in r["sells"] + r["buys"]:
            log(f"  {t['side']} {t['sym']} x{t['qty']} @ ${t['price']}")
        msg = build_message(r)
        sent = send_telegram(msg)
        log(f"טלגרם: {'נשלח ✓' if sent else 'לא נשלח'}")
    except Exception as e:
        log(f"שגיאה: {e}")
        send_telegram(f'⚠️ <b>מומנטום ארה"ב</b> — הריצה השבועית נכשלה: {e}')
        raise


if __name__ == "__main__":
    main()
