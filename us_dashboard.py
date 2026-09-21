# -*- coding: utf-8 -*-
"""
דשבורד מומנטום ארה"ב — Streamlit.

שני מצבי שימוש:
  1. עצמאי:   streamlit run us_dashboard.py --server.port 8502
  2. משובץ:   from us_dashboard import render_us_section
             render_us_section(mobile=...)   ← מוצג מתחת לדשבורד הישראלי
מציג את חשבון ה-IBKR (דמו/אמיתי) כתצוגה הראשית: משטר, שווי ופוזיציות בפועל,
עקומת הון מול המדדים, דירוג מומנטום, יישום ההחלטות ויומן החלטות.
תיק הנייר של הסוכן משמש רק כמנוע החלטות פנימי ואינו מוצג כ"תיק".
"""
from __future__ import annotations

import hashlib as _hashlib
import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE, "us_portfolio_state.json")


def _inject_us_css():
    st.markdown("""
    <style>
      .us-regime-bull { background:#0a3d1e; color:#4ade80; padding:12px 18px;
                        border-radius:10px; font-size:1.2rem; font-weight:bold; }
      .us-regime-bear { background:#450a0a; color:#f87171; padding:12px 18px;
                        border-radius:10px; font-size:1.2rem; font-weight:bold; }
      [data-testid="stMetricValue"] { direction: ltr; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=1800, show_spinner="טוען נתוני שוק אמריקאיים...")
def market_data():
    import yfinance as yf
    from backtest_us_v1 import UNIVERSE
    tickers = UNIVERSE + ["^GSPC", "^DJI", "^NDX"]
    px = yf.download(tickers, period="2y", interval="1d",
                     auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in px.index])
    return px.ffill()


def load_state():
    if not os.path.exists(STATE_FILE):
        return None
    with open(STATE_FILE, encoding="utf-8") as f:
        return json.load(f)


# ─── שער סיסמה (רק במצב עצמאי) — זהה לדשבורד הישראלי ──────────────────────────
_DASH_PW_HASH = "3090009d6533b344b3a4aae98c2133d3f394148243d9417f9f83dfd20d450182"


def _check_password() -> bool:
    expected_hash = _DASH_PW_HASH
    try:
        if "dashboard_password" in st.secrets:
            expected_hash = _hashlib.sha256(
                str(st.secrets["dashboard_password"]).encode()).hexdigest()
    except Exception:
        pass

    if st.session_state.get("_authed"):
        return True

    st.markdown("### 🔒 כניסה לדשבורד")
    pw = st.text_input("סיסמה", type="password", key="_pw_input")
    if st.button("כניסה"):
        if _hashlib.sha256((pw or "").encode()).hexdigest() == expected_hash:
            st.session_state["_authed"] = True
            st.rerun()
        else:
            st.error("סיסמה שגויה")
    st.stop()
    return False


def render_us_section(mobile: bool = False):
    """מציג את כל תוכן דשבורד ארה"ב. בטוח לקריאה מתוך אפליקציה אחרת
    (לא קורא ל-set_page_config ולא ל-st.stop())."""
    _inject_us_css()
    st.header('🇺🇸 מומנטום ארה"ב — רוטציה שבועית')

    state = load_state()
    if state is None:
        st.warning("עוד לא קיים תיק אמריקאי — הרץ פעם אחת את us_agent.py כדי לאתחל.")
        return
    try:
        px = market_data()
    except Exception as e:
        st.error(f"שגיאה בטעינת נתוני שוק אמריקאיים: {e}")
        return

    prices = px.iloc[-1]
    spx = px["^GSPC"].dropna(); ndx = px["^NDX"].dropna()
    spx_c = float(spx.iloc[-1]); ndx_c = float(ndx.iloc[-1])
    spx_dist = (spx_c / float(spx.rolling(200).mean().iloc[-1]) - 1) * 100
    ndx_dist = (ndx_c / float(ndx.rolling(200).mean().iloc[-1]) - 1) * 100
    bull = spx_dist > 0 and ndx_dist > 0   # משטר היברידי (v5b): שני המדדים מעל SMA200

    stocks = [c for c in px.columns if not c.startswith("^")]
    mom = (px[stocks].iloc[-21] / px[stocks].iloc[-252] - 1).dropna().sort_values(ascending=False)
    rank_of = {s: i + 1 for i, s in enumerate(mom.index)}

    # תיק הנייר של הסוכן — מנוע החלטות פנימי בלבד (נדרש למקדם ההון ולסטטוסים)
    held_val = sum(h["qty"] * float(prices[s]) for s, h in state["positions"].items()
                   if s in prices and pd.notna(prices[s]))
    pv = state["cash"] + held_val
    inc = state["inception"] or {"value": pv, "spx": spx_c, "dji": float(px["^DJI"].dropna().iloc[-1])}
    spx_cum = (spx_c / inc["spx"] - 1) * 100
    dji_cum = (float(px["^DJI"].dropna().iloc[-1]) / inc["dji"] - 1) * 100
    ndx_cum = (ndx_c / inc.get("ndx", ndx_c) - 1) * 100

    # ── חשבון IBKR — מקור האמת של המדידה ─────────────────────────────────────
    snap = None
    snap_path = os.path.join(BASE, "ibkr_snapshot.json")
    if os.path.exists(snap_path):
        with open(snap_path, encoding="utf-8") as f:
            snap = json.load(f)
    inc_eq = ((snap or {}).get("inception") or {}).get("equity") or (snap or {}).get("equity")
    cum_ib = (snap["equity"] / inc_eq - 1) * 100 if snap and inc_eq else 0.0

    # ── משטר + מדדים ──────────────────────────────────────────────────────────
    cls = "us-regime-bull" if bull else "us-regime-bear"
    detail = f"SPX ‏{spx_dist:+.1f}% · NDX ‏{ndx_dist:+.1f}% מול SMA200"
    txt = f"🐂 שוק שורי — {detail}" if bull else f"🐻 שוק דובי — {detail}"
    st.markdown(f'<div class="{cls}">{txt}</div>', unsafe_allow_html=True)
    acct_tag = ""
    if snap:
        acct_tag = f' · חשבון {"דמו 🧪" if snap.get("is_paper") else "אמיתי 💵"} {snap.get("account", "")}'
    st.caption("משטר היברידי (SPX+NDX) שבועי · ריבאלנס מומנטום חודשי · "
               f"Top6, buffer 14, מקס' מניה לסקטור{acct_tag}")

    if snap:
        metrics = [
            ("שווי חשבון IBKR", f'${snap["equity"]:,.0f}', f"{cum_ib:+.2f}%"),
            ("אלפא מול S&P", f"{cum_ib - spx_cum:+.2f}%", None),
            ("S&P 500 מההתחלה", f"{spx_cum:+.2f}%", None),
            ('נאסד"ק 100 מההתחלה', f"{ndx_cum:+.2f}%", None),
            ("דאו ג'ונס מההתחלה", f"{dji_cum:+.2f}%", None),
        ]
    else:
        metrics = [("חשבון IBKR", "אין נתונים", None),
                   ("S&P 500 מההתחלה", f"{spx_cum:+.2f}%", None)]
        st.warning("אין עדיין snapshot מ-IBKR — הרץ את us_ibkr_sync.py עם Gateway פתוח.")
    cols = st.columns(2 if mobile else 5)
    for i, (label, val, delta) in enumerate(metrics):
        cols[i % len(cols)].metric(label, val, delta)

    st.divider()

    from backtest_us_v6_next import SECTOR
    state_pos = state.get("positions", {})
    snap_pos = {p["sym"]: p for p in (snap or {}).get("positions", [])}
    today_str = datetime.now().strftime("%Y-%m-%d")
    trades_today = [t for t in state.get("trades", []) if t["date"] == today_str]

    # ── תיק פעיל — בסגנון הדשבורד הישראלי ─────────────────────────────────────
    REC = {
        "core":   ("#0a3d1e", "#4ade80", "החזק"),
        "buffer": ("#3d350a", "#fbbf24", "החזק — אזור החוצץ"),
        "edge":   ("#4a2c0a", "#fb923c", "זהירות — קרובה לגבול"),
        "sell":   ("#450a0a", "#f87171", "מכור בריבאלנס הקרוב"),
    }

    def _rec_for(sym):
        if not bull:
            return "sell", "שוק דובי — האסטרטגיה יוצאת מכל הפוזיציות לקרן כספית"
        rk = rank_of.get(sym)
        if rk is None:
            return "sell", "לא בדירוג (חסרים נתונים) — תימכר בריבאלנס הקרוב"
        if rk <= 6:
            return "core", f"דירוג #{rk} — בליבת הטופ-6, ממשיכים להחזיק"
        if rk <= 12:
            return "buffer", f"דירוג #{rk} — בתוך החוצץ; מכירה רק בנפילה מתחת לדירוג 14"
        if rk <= 14:
            return "edge", f"דירוג #{rk} — צעד אחד מגבול המכירה (14); תימכר אם תמשיך לרדת"
        return "sell", f"דירוג #{rk} — מתחת לגבול 14; תימכר בריבאלנס החודשי הקרוב"

    if snap_pos:
        st.subheader(f"💼 תיק פעיל ({len(snap_pos)} פוזיציות)")
        total_val = sum((p.get("value") or 0) for p in snap_pos.values()) + (snap.get("cash") or 0)
        for sym in sorted(snap_pos, key=lambda s: rank_of.get(s, 999)):
            p_ = snap_pos[sym]
            key, reason = _rec_for(sym)
            bg, fg, rec_he = REC[key]
            avg = p_.get("avg_cost") or 0
            cur = p_.get("price") or 0
            pnl_pct = p_.get("pnl_pct")
            pnl_usd = p_.get("pnl")
            buy_date = state_pos.get(sym, {}).get("buy_date", "—")
            weight = (p_.get("value") or 0) / total_val * 100 if total_val else 0
            mom_val = mom.get(sym)
            rk = rank_of.get(sym, "—")
            pnl_txt = f"{pnl_pct:+.1f}%" if pnl_pct is not None else "—"
            header = (f'{sym} | כניסה: ${avg:,.2f} ({buy_date}) | '
                      f'כעת: ${cur:,.2f} | רו"ה: {pnl_txt} | {rec_he}')
            with st.expander(header, expanded=(key == "sell")):
                st.markdown(
                    f'<div style="background:{bg};color:{fg};padding:10px 16px;'
                    f'border-radius:6px;font-weight:bold;font-size:1.05rem;">'
                    f'{rec_he} — {reason}</div>', unsafe_allow_html=True)
                st.markdown("")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**מחירים ורווח**")
                    st.markdown(f"""
| פרמטר | ערך |
|-------|-----|
| מחיר כניסה | ${avg:,.2f} |
| מחיר נוכחי | ${cur:,.2f} |
| רו"ה % | {pnl_txt} |
| רו"ה $ | {f'${pnl_usd:+,.0f}' if pnl_usd is not None else '—'} |
""")
                with c2:
                    st.markdown("**הפוזיציה**")
                    st.markdown(f"""
| פרמטר | ערך |
|-------|-----|
| כמות | {p_.get('qty', '—'):,} |
| שווי | ${p_.get('value') or 0:,.0f} |
| משקל בתיק | {weight:.1f}% |
| סקטור | {SECTOR.get(sym, '—')} |
""")
                with c3:
                    st.markdown("**דירוג מומנטום**")
                    room = (14 - rk) if isinstance(rk, int) else "—"
                    st.markdown(f"""
| פרמטר | ערך |
|-------|-----|
| דירוג נוכחי | #{rk} |
| מומנטום 12-1 | {f'{mom_val*100:+.1f}%' if mom_val is not None else '—'} |
| גבול מכירה | דירוג 14 |
| מרווח עד מכירה | {room} דירוגים |
""")

        # הרכב התיק — משקלים
        comp = {sym: (p.get("value") or 0) / total_val * 100
                for sym, p in snap_pos.items() if total_val}
        if snap.get("cash"):
            comp["מזומן"] = snap["cash"] / total_val * 100
        st.markdown("**הרכב התיק (%)**")
        st.bar_chart(pd.Series(comp).sort_values(ascending=False), height=180)
        st.caption(f'💵 מזומן בחשבון: ${(snap.get("cash") or 0):,.0f} · '
                   f'עדכון אחרון: {snap.get("updated", "—")}')
    elif snap:
        st.info("אין פוזיציות בחשבון — התיק בקרן כספית (שוק דובי).")
    else:
        st.info("ההחזקות יוצגו אחרי הסנכרון הראשון מ-IBKR.")

    # ── המלצות קנייה ומכירה — כמו בדשבורד הישראלי ─────────────────────────────
    buys_today = [t for t in trades_today if t["side"] == "BUY"]
    sells_today = [t for t in trades_today if t["side"] == "SELL"]

    def _buy_section():
        st.subheader(f"🟢 איתותי קנייה ({len(buys_today)})")
        if buys_today:
            for t in buys_today:
                st.success(f'**{t["sym"]}** (דירוג #{t.get("rank", "?")}) — '
                           f'{t["qty"]} יח׳ @ ${t["price"]:,.2f} · {t.get("reason", "")}')
        elif not bull:
            st.info("שוק דובי — האסטרטגיה לא קונה מניות; ההון בקרן כספית.")
        else:
            st.info("אין קניות השבוע — כל 6 הסלוטים מאוישים והריבאלנס הבא בתחילת החודש.")
        # הבאות בתור — מי תיכנס אם יתפנה מקום (בכפוף לתקרת הסקטור)
        if bull:
            held_secs = {SECTOR.get(s, "?") for s in snap_pos}
            queue = []
            for s in mom.index:
                if s in snap_pos:
                    continue
                sec = SECTOR.get(s, "?")
                if sec in held_secs:
                    continue
                queue.append(f"{s} (#{rank_of[s]}, {sec})")
                held_secs.add(sec)
                if len(queue) == 3:
                    break
            if queue:
                st.caption("הבאות בתור אם יתפנה סלוט: " + " · ".join(queue))

    def _sell_section():
        at_risk = [(s, rank_of.get(s)) for s in snap_pos
                   if isinstance(rank_of.get(s), int) and rank_of[s] > 14]
        near = [(s, rank_of.get(s)) for s in snap_pos
                if isinstance(rank_of.get(s), int) and 13 <= rank_of[s] <= 14]
        st.subheader(f"🔴 איתותי מכירה / זהירות ({len(sells_today) + len(at_risk)})")
        if sells_today:
            for t in sells_today:
                st.error(f'**{t["sym"]}** — {t["qty"]} יח׳ @ ${t["price"]:,.2f} '
                         f'| {t.get("pnl_pct", 0):+.1f}% | {t.get("reason", "")}')
        for s, rk in at_risk:
            st.error(f'**{s}** — דירוג #{rk}, מתחת לגבול 14: תימכר בריבאלנס החודשי הקרוב.')
        for s, rk in near:
            st.warning(f'**{s}** — דירוג #{rk}, צמוד לגבול המכירה. במעקב.')
        if not sells_today and not at_risk and not near:
            st.info("אין מועמדות למכירה — כל ההחזקות עמוק בתוך החוצץ.")

    if mobile:
        _buy_section()
        _sell_section()
    else:
        cb, cs = st.columns(2)
        with cb: _buy_section()
        with cs: _sell_section()

    st.divider()

    def _equity_and_portfolio():
        st.subheader("📈 החשבון מול המדדים")
        if snap and len(snap.get("history", [])) >= 2 and inc_eq:
            hi = pd.DataFrame(snap["history"])
            hi["date"] = pd.to_datetime(hi["date"])
            hi = hi.set_index("date").sort_index()
            series = {"חשבון IBKR": hi["equity"] / inc_eq * 100 - 100}
            for label, tk, base in (("S&P 500", "^GSPC", inc.get("spx")),
                                    ('נאסד"ק 100', "^NDX", inc.get("ndx")),
                                    ("דאו ג'ונס", "^DJI", inc.get("dji"))):
                if base:
                    b = px[tk].reindex(hi.index, method="ffill")
                    series[label] = b / base * 100 - 100
            st.line_chart(pd.DataFrame(series), height=340)
        else:
            st.info("עקומת ההון תופיע אחרי כמה סנכרונים יומיים מ-IBKR.")

    def _momentum():
        st.subheader("🏁 דירוג מומנטום 12-1")
        st.caption("ירוק = מוחזק · צהוב = באזור החוצץ (7–14) · מכירה מתחת לדירוג 14 · מקסימום מניה אחת לסקטור")
        from backtest_us_v6_next import SECTOR
        rows = []
        for i, (s, m) in enumerate(mom.head(20).items(), start=1):
            held = s in state["positions"]
            zone = "✅ מוחזק" if held else ("🎯 טופ-6" if i <= 6 else ("🟡 חוצץ" if i <= 14 else ""))
            rows.append({"#": i, "מניה": s, "סקטור": SECTOR.get(s, "—"),
                         "מומנטום 12-1": f"{m*100:+.1f}%", "סטטוס": zone})
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True,
                     height=None if mobile else 560)

    if mobile:
        _equity_and_portfolio()
        _momentum()
    else:
        col_l, col_r = st.columns([3, 2])
        with col_l: _equity_and_portfolio()
        with col_r: _momentum()

    # ── יישום החלטות האסטרטגיה בחשבון ─────────────────────────────────────────
    if snap:
        st.divider()
        st.subheader("🔀 יישום החלטות האסטרטגיה בחשבון")
        st.caption("השוואה בין הרכב היעד שהאסטרטגיה קבעה לבין הפוזיציות בפועל ב-IBKR")
        agent_val = state["history"][-1]["value"] if state.get("history") else None
        if agent_val and snap.get("equity") and state.get("positions"):
            scale = snap["equity"] / agent_val

            ib_qty = {p["sym"]: p["qty"] for p in snap.get("positions", [])}
            rows, worst = [], 0.0
            for sym, h in state["positions"].items():
                target = h["qty"] * scale
                actual = ib_qty.get(sym, 0)
                dev = (actual / target - 1) * 100 if target else 0.0
                worst = max(worst, abs(dev))
                status = "✅" if abs(dev) <= 7 else "⚠️ פער"
                rows.append({"מניה": sym, "יעד (סוכן ×מקדם)": f"{target:,.0f}",
                             "בפועל ב-IBKR": f"{actual:,.0f}",
                             "סטייה": f"{dev:+.1f}%", "סטטוס": status})
            for sym in ib_qty:
                if sym not in state["positions"]:
                    rows.append({"מניה": sym, "יעד (סוכן ×מקדם)": "0",
                                 "בפועל ב-IBKR": f"{ib_qty[sym]:,.0f}",
                                 "סטייה": "—", "סטטוס": "⚠️ עודף"})
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            n_issues = sum(1 for r in rows if "⚠️" in r["סטטוס"])
            if n_issues:
                st.warning(f"{n_issues} פוזיציות חורגות מהיעד — הרץ יישור: "
                           f"us_executor.py --sync-target")
            else:
                st.success(f"כל הפוזיציות בתחום הסבילות (עד 7%) · מקדם הון x{scale:.2f}")
        else:
            st.info("ההשוואה תוצג כשיש גם תיק סוכן וגם snapshot מ-IBKR.")

    # ── יומן עסקאות ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📜 יומן החלטות האסטרטגיה")
    if state["trades"]:
        dft = pd.DataFrame(state["trades"])[::-1].rename(columns={
            "date": "תאריך", "side": "פעולה", "sym": "מניה", "qty": "כמות",
            "price": "מחיר $", "pnl_pct": "רווח %", "tax": "מס $",
            "rank": "דירוג", "reason": "סיבה"})
        st.dataframe(dft, width='stretch', hide_index=True)
    else:
        st.info("אין עסקאות עדיין.")
    st.caption(f"עודכן: {datetime.now():%Y-%m-%d %H:%M} · המדידה על חשבון IBKR · אינו ייעוץ השקעות")


# ─── מצב עצמאי בלבד ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(page_title='מומנטום ארה"ב', page_icon="🇺🇸", layout="wide")
    st.markdown("<style>.stApp{direction:rtl;} h1,h2,h3,p,div,span{text-align:right;}</style>",
                unsafe_allow_html=True)
    _check_password()
    render_us_section(mobile=False)
