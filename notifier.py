# -*- coding: utf-8 -*-
"""
שולח איתותי מסחר יומיים למייל.

קונפיגורציה: קובץ מקומי email_config.json (לא נשמר בגיטהאב). פורמט:
{
  "enabled": true,
  "smtp_host": "smtp.gmail.com",
  "smtp_port": 587,
  "sender": "yanivbarshaf@gmail.com",
  "app_password": "xxxx xxxx xxxx xxxx",   <- App Password של Gmail (לא הסיסמה הרגילה)
  "recipient": "yanivbarshaf@gmail.com"
}

אם הקובץ חסר / enabled=false / אין app_password — הפונקציה לא עושה כלום (no-op),
כך שהמנוע ממשיך לעבוד גם בלי הגדרת מייל.
"""
from __future__ import annotations

import json
import os
import smtplib
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from securities import hname

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "email_config.json")


def _load_config() -> dict | None:
    if not os.path.exists(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"  [notifier] שגיאה בקריאת email_config.json: {e}")
        return None
    if not cfg.get("enabled"):
        return None
    if not cfg.get("app_password") or not cfg.get("sender") or not cfg.get("recipient"):
        print("  [notifier] email_config.json חסר sender/app_password/recipient — לא נשלח מייל")
        return None
    cfg.setdefault("smtp_host", "smtp.gmail.com")
    cfg.setdefault("smtp_port", 587)
    return cfg


def _fmt_pct(v: float) -> str:
    return f"{v:+.2f}%"


def _build_html(payload: dict) -> str:
    d        = payload["date"]
    regime   = payload["regime"]
    port_val = payload["portfolio_val"]
    cum_pct  = payload["cum_pct"]
    cash     = payload["cash"]
    bonds    = payload["bonds"]
    equity   = payload["equity"]
    entries  = payload["entries"]   # [{sym, base_price, pos_pct, rs, qty_total, tranches}, ...]
    exits    = payload["exits"]     # [(sym, reason, pnl_pct), ...]
    holdings = payload["holdings"]  # [(sym, qty, entry, cur, pnl_pct), ...]
    projected = payload.get("projected")          # תיק יעד אחרי ביצוע
    bond_instruments = payload.get("bond_instruments", [])

    regime_he = {"BULL": "שורי 🐂", "NEUTRAL": "ניטרלי", "BEAR": "דובי 🐻"}.get(regime, regime)

    def row(cells, header=False):
        tag = "th" if header else "td"
        style = "padding:8px 10px;border-bottom:1px solid #e5e7eb;text-align:right;"
        if header:
            style += "background:#1f2937;color:#fff;font-weight:bold;"
        return "<tr>" + "".join(f"<{tag} style='{style}'>{c}</{tag}>" for c in cells) + "</tr>"

    # ── BUY section — כמות, טווח מחירים וכניסה מדורגת ──
    if entries:
        buy_rows = ""
        for e in entries:
            s = e["sym"]; base = e["base_price"]; qty = e["qty_total"]
            tranches = e["tranches"]   # [(tranche_no, limit_price, qty), ...]
            # טווח מחירים: מהמחיר הנמוך (טרנש אחרון) עד מחיר השוק (טרנש 1)
            prices = [lp for (_t, lp, _q) in tranches]
            lo, hi = (min(prices), max(prices)) if prices else (base, base)
            ladder_txt = " · ".join(
                f"T{t}: {q:,.0f} יח׳ @ ₪{lp:,.0f}" + (" (שוק)" if t == 1 else " (לימיט)")
                for (t, lp, q) in tranches
            )
            buy_rows += row([
                hname(s),
                f"<b>{qty:,.0f} יח׳</b>",
                f"₪{lo:,.0f} – ₪{hi:,.0f}",
                f"{e['pos_pct']:.1f}%",
                _fmt_pct(e["rs"]),
            ])
            buy_rows += (f"<tr><td colspan='5' style='padding:4px 14px 10px;"
                         f"color:#374151;font-size:12px;background:#f9fafb;'>↳ {ladder_txt}</td></tr>")
        buy_html = f"""
        <h3 style="color:#15803d;">🟢 המלצות קנייה ({len(entries)})</h3>
        <table style="border-collapse:collapse;width:100%;font-size:14px;">
          {row(['נייר', 'כמות (יח׳)', 'טווח מחירים', '% מהתיק', 'RS מול מדד'], header=True)}
          {buy_rows}
        </table>
        <p style="font-size:12px;color:#6b7280;">
          כניסה מדורגת (Ladder): T1 = מחיר שוק · T2 = ~1%- · T3 = ~2%- מתחת. הכל ביחידות שלמות.
        </p>"""
    else:
        buy_html = '<h3 style="color:#15803d;">🟢 המלצות קנייה</h3><p>אין כניסות חדשות היום.</p>'

    # ── SELL section — כמות, טווח מחיר מכירה וסיבה ──
    reason_he = {"trail_stop": "Trail Stop", "take_profit": "Take Profit",
                 "signal_exit": "יציאת איתות", "regime_bear": "שוק דובי"}
    if exits:
        sell_rows = "".join(
            row([
                hname(e["sym"]),
                f"<b>{e['qty']:,.0f} יח׳</b>",
                f"₪{e['sell_low']:,.0f} – ₪{e['sell_high']:,.0f}",
                reason_he.get(e["reason"], e["reason"]),
                _fmt_pct(e["pnl_pct"]),
            ])
            for e in exits
        )
        sell_html = f"""
        <h3 style="color:#b91c1c;">🔴 המלצות מכירה ({len(exits)})</h3>
        <table style="border-collapse:collapse;width:100%;font-size:14px;">
          {row(['נייר', 'כמות (יח׳)', 'טווח מחיר מכירה', 'סיבה', 'רווח/הפסד'], header=True)}
          {sell_rows}
        </table>
        <p style="font-size:12px;color:#6b7280;">
          טווח המכירה מחושב סביב מחיר הסגירה האחרון (יעד = מחיר שוק, רצפה מקובלת ~1%- כדי להבטיח יציאה).
          ביציאת Trail Stop מומלצת פקודת שוק.
        </p>"""
    else:
        sell_html = '<h3 style="color:#b91c1c;">🔴 המלצות מכירה</h3><p>אין יציאות היום.</p>'

    # ── Holdings ──
    if holdings:
        hold_rows = "".join(
            row([hname(s), f"{qty:,.0f}", f"₪{entry:,.0f}", f"₪{cur:,.0f}", _fmt_pct(pnl)])
            for (s, qty, entry, cur, pnl) in holdings
        )
        hold_html = f"""
        <h3>📊 החזקות מניות נוכחיות</h3>
        <table style="border-collapse:collapse;width:100%;font-size:14px;">
          {row(['נייר', 'כמות', 'מחיר כניסה', 'מחיר נוכחי', 'רווח/הפסד'], header=True)}
          {hold_rows}
        </table>"""
    else:
        hold_html = "<h3>📊 החזקות מניות נוכחיות</h3><p>אין פוזיציות מניות פתוחות.</p>"

    # ── Recommended portfolio AFTER executing today's actions ──
    if projected:
        tot = projected["total"] or 1
        comp_rows = ""
        for (s, qty, val) in projected["stocks"]:
            comp_rows += row([hname(s), f"{qty:,.0f} יח׳", f"₪{val:,.0f}", f"{val/tot*100:.1f}%"])
        # אג"ח (מצרפי) + פירוט מכשירים
        if projected["bonds"] > 0 and bond_instruments:
            split = projected["bonds"] / len(bond_instruments)
            for (name, secn) in bond_instruments:
                comp_rows += row([f'{name} · נ"ע {secn}', "—",
                                  f"₪{split:,.0f}", f"{split/tot*100:.1f}%"])
        elif projected["bonds"] > 0:
            comp_rows += row(['אג"ח ממשלתי', "—", f"₪{projected['bonds']:,.0f}",
                              f"{projected['bonds']/tot*100:.1f}%"])
        comp_rows += row(["מזומן", "—", f"₪{projected['cash']:,.0f}",
                          f"{projected['cash']/tot*100:.1f}%"])
        proj_html = f"""
        <h3 style="color:#1d4ed8;">🎯 הרכב התיק המומלץ לאחר ביצוע הפעולות</h3>
        <table style="border-collapse:collapse;width:100%;font-size:14px;">
          {row(['רכיב', 'כמות', 'שווי', '% מהתיק'], header=True)}
          {comp_rows}
          {row([f'<b>סה"כ</b>', '', f'<b>₪{projected["total"]:,.0f}</b>', '<b>100%</b>'])}
        </table>
        <p style="font-size:12px;color:#6b7280;">
          מניות ₪{projected['equity']:,.0f} · אג"ח ₪{projected['bonds']:,.0f} · מזומן ₪{projected['cash']:,.0f}.
          מבוסס על הנחת מילוי מלא של פקודות הקנייה במחיר הבסיס.
        </p>"""
    else:
        proj_html = ""

    return f"""<!DOCTYPE html>
<html dir="rtl" lang="he">
<head><meta charset="utf-8"></head>
<body style="direction:rtl;text-align:right;font-family:'Segoe UI',Arial,sans-serif;
             color:#111827;max-width:680px;margin:auto;">
  <h2 style="border-bottom:3px solid #2563eb;padding-bottom:6px;">
    איתותי מסחר ת"א 125 — {d}
  </h2>
  <div style="background:#f3f4f6;border-radius:8px;padding:14px;margin:12px 0;">
    <b>מצב שוק (Regime):</b> {regime_he}<br>
    <b>שווי תיק:</b> ₪{port_val:,.0f} &nbsp;|&nbsp;
    <b>תשואה מצטברת:</b> {_fmt_pct(cum_pct)}<br>
    <b>מניות:</b> ₪{equity:,.0f} &nbsp;|&nbsp;
    <b>אג"ח:</b> ₪{bonds:,.0f} &nbsp;|&nbsp;
    <b>מזומן:</b> ₪{cash:,.0f}
  </div>
  {buy_html}
  <br>
  {sell_html}
  <br>
  {proj_html}
  <br>
  {hold_html}
  <p style="color:#6b7280;font-size:12px;margin-top:20px;border-top:1px solid #e5e7eb;padding-top:10px;">
    הודעה אוטומטית ממערכת ה-Paper Trading. אינה מהווה ייעוץ השקעות.
    הביצוע בפועל באחריותך, דרך הברוקר שלך.
  </p>
</body>
</html>"""


def send_signals_email(payload: dict) -> bool:
    """שולח את איתותי היום למייל. מחזיר True אם נשלח, False אם דולג/נכשל."""
    cfg = _load_config()
    if cfg is None:
        return False

    try:
        msg = MIMEMultipart("alternative")
        n_buy = len(payload.get("entries", []))
        n_sell = len(payload.get("exits", []))
        msg["Subject"] = f'איתותי ת"א 125 {payload["date"]} — {n_buy} קניות, {n_sell} מכירות'
        msg["From"] = cfg["sender"]
        msg["To"] = cfg["recipient"]
        msg.attach(MIMEText(_build_html(payload), "html", "utf-8"))

        ctx = ssl.create_default_context()
        with smtplib.SMTP(cfg["smtp_host"], int(cfg["smtp_port"]), timeout=30) as server:
            server.starttls(context=ctx)
            server.login(cfg["sender"], cfg["app_password"])
            server.sendmail(cfg["sender"], [cfg["recipient"]], msg.as_string())
        print(f"  [notifier] מייל איתותים נשלח אל {cfg['recipient']}")
        return True
    except Exception as e:
        print(f"  [notifier] שליחת מייל נכשלה: {e}")
        return False


if __name__ == "__main__":
    # בדיקת שליחה עם נתוני דמה
    demo = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "regime": "NEUTRAL", "portfolio_val": 98443, "cum_pct": -1.56,
        "cash": 20000, "bonds": 40060, "equity": 38383,
        "entries": [{
            "sym": "POLI.TA", "base_price": 7250, "pos_pct": 8.0, "rs": 3.2,
            "qty_total": 11,
            "tranches": [(1, 7250, 6), (2, 7178, 3), (3, 7105, 2)],
        }],
        "exits": [{"sym": "TEVA.TA", "reason": "trail_stop", "pnl_pct": -4.1,
                   "qty": 8, "price": 1820, "sell_low": 1802, "sell_high": 1820}],
        "holdings": [("RSEL.TA", 5, 1110, 1125, 1.35),
                     ("AMOT.TA", 10, 1876, 1880, 0.21)],
        "projected": {
            "total": 98443, "equity": 78383, "bonds": 40060, "cash": 0,
            "stocks": [("POLI.TA", 11, 79750), ("RSEL.TA", 5, 5625), ("AMOT.TA", 10, 18800)],
        },
        "bond_instruments": [("ממשלתי שקלי 0327 — שקלי קצר", "1139344"),
                             ("ממשלתי שקלי 0928 — שקלי בינוני", "1150879")],
    }
    ok = send_signals_email(demo)
    print("נשלח" if ok else "לא נשלח (בדוק email_config.json)")
