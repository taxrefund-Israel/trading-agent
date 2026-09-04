# -*- coding: utf-8 -*-
"""
דשבורד מומנטום ארה"ב — Streamlit.

שני מצבי שימוש:
  1. עצמאי:   streamlit run us_dashboard.py --server.port 8502
  2. משובץ:   from us_dashboard import render_us_section
             render_us_section(mobile=...)   ← מוצג מתחת לדשבורד הישראלי
מציג: מצב משטר, התיק הנוכחי, דירוג מומנטום מלא, עקומת הון מול המדדים, יומן עסקאות.
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

    held_val = sum(h["qty"] * float(prices[s]) for s, h in state["positions"].items()
                   if s in prices and pd.notna(prices[s]))
    pv = state["cash"] + held_val
    inc = state["inception"] or {"value": pv, "spx": spx_c, "dji": float(px["^DJI"].dropna().iloc[-1])}
    cum = (pv / inc["value"] - 1) * 100
    spx_cum = (spx_c / inc["spx"] - 1) * 100
    dji_cum = (float(px["^DJI"].dropna().iloc[-1]) / inc["dji"] - 1) * 100
    ndx_cum = (ndx_c / inc.get("ndx", ndx_c) - 1) * 100

    # ── משטר + מדדים ──────────────────────────────────────────────────────────
    cls = "us-regime-bull" if bull else "us-regime-bear"
    detail = f"SPX ‏{spx_dist:+.1f}% · NDX ‏{ndx_dist:+.1f}% מול SMA200"
    txt = f"🐂 שוק שורי — {detail}" if bull else f"🐻 שוק דובי — {detail}"
    st.markdown(f'<div class="{cls}">{txt}</div>', unsafe_allow_html=True)
    st.caption("משטר היברידי (SPX+NDX) שבועי · ריבאלנס מומנטום חודשי · Top6, buffer 14, מקס' מניה לסקטור")

    metrics = [
        ("שווי התיק", f"${pv:,.0f}", f"{cum:+.2f}%"),
        ("אלפא מול S&P", f"{cum - spx_cum:+.2f}%", None),
        ("S&P 500 מההתחלה", f"{spx_cum:+.2f}%", None),
        ('נאסד"ק 100 מההתחלה', f"{ndx_cum:+.2f}%", None),
        ("דאו ג'ונס מההתחלה", f"{dji_cum:+.2f}%", None),
    ]
    cols = st.columns(2 if mobile else 5)
    for i, (label, val, delta) in enumerate(metrics):
        cols[i % len(cols)].metric(label, val, delta)

    st.divider()

    def _equity_and_portfolio():
        st.subheader("📈 התיק מול המדדים")
        hist = pd.DataFrame(state["history"])
        if len(hist) >= 2:
            hist["date"] = pd.to_datetime(hist["date"]); hist = hist.set_index("date")
            series = {
                "התיק":    hist["value"] / inc["value"] * 100 - 100,
                "S&P 500": hist["spx"] / inc["spx"] * 100 - 100,
                "דאו ג'ונס": hist["dji"] / inc["dji"] * 100 - 100,
            }
            if "ndx" in hist.columns and "ndx" in inc:
                series['נאסד"ק 100'] = hist["ndx"] / inc["ndx"] * 100 - 100
            st.line_chart(pd.DataFrame(series), height=340)
        else:
            st.info("עקומת ההון תופיע אחרי כמה ריצות שבועיות.")

        st.subheader("🎯 התיק הנוכחי")
        if state["positions"]:
            rows = []
            for s, h in state["positions"].items():
                p = float(prices[s]) if s in prices and pd.notna(prices[s]) else None
                avg = h["cost"] / h["qty"]
                rows.append({
                    "מניה": s, "דירוג": rank_of.get(s, "—"), "כמות": h["qty"],
                    "מחיר קנייה": round(avg, 2), "מחיר נוכחי": round(p, 2) if p else "—",
                    "שווי $": round(h["qty"] * p, 0) if p else "—",
                    "רווח %": round((p / avg - 1) * 100, 1) if p else "—",
                    "נקנתה": h.get("buy_date", "—"),
                })
            st.dataframe(pd.DataFrame(rows).sort_values("דירוג"),
                         width='stretch', hide_index=True)
            st.caption(f'💵 מזומן/קרן כספית: ${state["cash"]:,.0f} · '
                       f'מס ששולם מצטבר: ${state.get("tax_paid", 0):,.0f}')
        else:
            st.info("אין פוזיציות — התיק בקרן כספית (שוק דובי) או טרם אותחל.")

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

    # ── IBKR בפועל ─────────────────────────────────────────────────────────────
    snap_path = os.path.join(BASE, "ibkr_snapshot.json")
    if os.path.exists(snap_path):
        with open(snap_path, encoding="utf-8") as f:
            snap = json.load(f)
        st.divider()
        acct_tag = "🧪 דמו" if snap.get("is_paper") else "💵 אמיתי"
        st.subheader(f'🏦 חשבון IBKR בפועל ({acct_tag} {snap.get("account", "")})')
        inc_eq = (snap.get("inception") or {}).get("equity") or snap["equity"]
        cum_ib = (snap["equity"] / inc_eq - 1) * 100 if inc_eq else 0.0
        i1, i2, i3 = st.columns(3)
        i1.metric("שווי חשבון", f'${snap["equity"]:,.0f}', f"{cum_ib:+.2f}%")
        i2.metric("מזומן", f'${snap.get("cash") or 0:,.0f}')
        i3.metric("עדכון אחרון", snap.get("updated", "—"))
        if snap.get("positions"):
            dfi = pd.DataFrame(snap["positions"]).rename(columns={
                "sym": "מניה", "qty": "כמות", "avg_cost": "עלות ממוצעת",
                "price": "מחיר", "value": "שווי $", "pnl": 'רו"ה $', "pnl_pct": 'רו"ה %'})
            st.dataframe(dfi, width='stretch', hide_index=True)
        if len(snap.get("history", [])) >= 2:
            hi = pd.DataFrame(snap["history"])
            hi["date"] = pd.to_datetime(hi["date"])
            st.line_chart(hi.set_index("date")["equity"], height=200)
        st.caption("הנתונים נקראים ישירות מחשבון IBKR (מתעדכן בהרצת us_ibkr_sync).")

    # ── יומן עסקאות ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📜 יומן עסקאות")
    if state["trades"]:
        dft = pd.DataFrame(state["trades"])[::-1].rename(columns={
            "date": "תאריך", "side": "פעולה", "sym": "מניה", "qty": "כמות",
            "price": "מחיר $", "pnl_pct": "רווח %", "tax": "מס $",
            "rank": "דירוג", "reason": "סיבה"})
        st.dataframe(dft, width='stretch', hide_index=True)
    else:
        st.info("אין עסקאות עדיין.")
    st.caption(f"עודכן: {datetime.now():%Y-%m-%d %H:%M} · תיק נייר · אינו ייעוץ השקעות")


# ─── מצב עצמאי בלבד ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(page_title='מומנטום ארה"ב', page_icon="🇺🇸", layout="wide")
    st.markdown("<style>.stApp{direction:rtl;} h1,h2,h3,p,div,span{text-align:right;}</style>",
                unsafe_allow_html=True)
    _check_password()
    render_us_section(mobile=False)
