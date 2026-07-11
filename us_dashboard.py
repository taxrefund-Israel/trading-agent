# -*- coding: utf-8 -*-
"""
דשבורד מומנטום ארה"ב — Streamlit
הרצה: streamlit run us_dashboard.py --server.port 8502
מציג: מצב משטר, התיק הנוכחי, דירוג מומנטום מלא, עקומת הון מול המדדים, יומן עסקאות.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE, "us_portfolio_state.json")

st.set_page_config(page_title='מומנטום ארה"ב', page_icon="🇺🇸", layout="wide")

st.markdown("""
<style>
  .stApp { direction: rtl; }
  h1, h2, h3, p, div, span { text-align: right; }
  .regime-bull { background:#0a3d1e; color:#4ade80; padding:12px 18px;
                 border-radius:10px; font-size:1.2rem; font-weight:bold; }
  .regime-bear { background:#450a0a; color:#f87171; padding:12px 18px;
                 border-radius:10px; font-size:1.2rem; font-weight:bold; }
  [data-testid="stMetricValue"] { direction: ltr; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=1800, show_spinner="טוען נתוני שוק...")
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


state = load_state()
st.title('🇺🇸 מומנטום ארה"ב — רוטציה שבועית')

if state is None:
    st.warning("עוד לא קיים תיק — הרץ פעם אחת את us_agent.py כדי לאתחל.")
    st.stop()

px = market_data()
prices = px.iloc[-1]
spx = px["^GSPC"].dropna()
ndx = px["^NDX"].dropna()
spx_c = float(spx.iloc[-1])
ndx_c = float(ndx.iloc[-1])
spx_dist = (spx_c / float(spx.rolling(200).mean().iloc[-1]) - 1) * 100
ndx_dist = (ndx_c / float(ndx.rolling(200).mean().iloc[-1]) - 1) * 100
# משטר היברידי (v5b): שורי רק כששני המדדים מעל SMA200
bull = spx_dist > 0 and ndx_dist > 0

stocks = [c for c in px.columns if not c.startswith("^")]
mom = (px[stocks].iloc[-21] / px[stocks].iloc[-252] - 1).dropna().sort_values(ascending=False)
rank_of = {s: i + 1 for i, s in enumerate(mom.index)}

# ─── שורה עליונה: משטר + מדדים ────────────────────────────────────────────────
held_val = sum(h["qty"] * float(prices[s]) for s, h in state["positions"].items()
               if s in prices and pd.notna(prices[s]))
pv = state["cash"] + held_val
inc = state["inception"] or {"value": pv, "spx": spx_c, "dji": float(px["^DJI"].dropna().iloc[-1])}
cum = (pv / inc["value"] - 1) * 100
spx_cum = (spx_c / inc["spx"] - 1) * 100
dji_cum = (float(px["^DJI"].dropna().iloc[-1]) / inc["dji"] - 1) * 100
ndx_cum = (ndx_c / inc.get("ndx", ndx_c) - 1) * 100

c1, c2, c3, c4, c5, c6 = st.columns([2, 1, 1, 1, 1, 1])
with c1:
    cls = "regime-bull" if bull else "regime-bear"
    detail = f"SPX ‏{spx_dist:+.1f}% · NDX ‏{ndx_dist:+.1f}% מול SMA200"
    txt = f"🐂 שוק שורי — {detail}" if bull else f"🐻 שוק דובי — {detail}"
    st.markdown(f'<div class="{cls}">{txt}</div>', unsafe_allow_html=True)
    st.caption("משטר היברידי (SPX+NDX) שבועי · ריבאלנס מומנטום: חודשי · Top5, buffer 12")
c2.metric("שווי התיק", f"${pv:,.0f}", f"{cum:+.2f}%")
c3.metric("אלפא מול S&P", f"{cum - spx_cum:+.2f}%")
c4.metric("S&P 500 מאז ההתחלה", f"{spx_cum:+.2f}%")
c5.metric('נאסד"ק 100 מאז ההתחלה', f"{ndx_cum:+.2f}%")
c6.metric("דאו ג'ונס מאז ההתחלה", f"{dji_cum:+.2f}%")

st.divider()
col_l, col_r = st.columns([3, 2])

# ─── עקומת הון ────────────────────────────────────────────────────────────────
with col_l:
    st.subheader("📈 התיק מול המדדים")
    hist = pd.DataFrame(state["history"])
    if len(hist) >= 2:
        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.set_index("date")
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
        dfp = pd.DataFrame(rows).sort_values("דירוג")
        st.dataframe(dfp, width='stretch', hide_index=True)
        st.caption(f'💵 מזומן/קרן כספית: ${state["cash"]:,.0f} · '
                   f'מס ששולם מצטבר: ${state.get("tax_paid", 0):,.0f}')
    else:
        st.info("אין פוזיציות — התיק בקרן כספית (שוק דובי) או טרם אותחל.")

# ─── דירוג מומנטום ────────────────────────────────────────────────────────────
with col_r:
    st.subheader("🏁 דירוג מומנטום 12-1")
    st.caption("ירוק = מוחזק · צהוב = באזור החוצץ (6–12) · מכירה רק מתחת לדירוג 12")
    top = mom.head(20)
    rows = []
    for i, (s, m) in enumerate(top.items(), start=1):
        held = s in state["positions"]
        zone = "✅ מוחזק" if held else ("🎯 טופ-5" if i <= 5 else ("🟡 חוצץ" if i <= 12 else ""))
        rows.append({"#": i, "מניה": s, "מומנטום 12-1": f"{m*100:+.1f}%", "סטטוס": zone})
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True, height=560)

# ─── יומן עסקאות ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("📜 יומן עסקאות")
if state["trades"]:
    dft = pd.DataFrame(state["trades"])[::-1]
    dft = dft.rename(columns={"date": "תאריך", "side": "פעולה", "sym": "מניה",
                              "qty": "כמות", "price": "מחיר $", "pnl_pct": "רווח %",
                              "tax": "מס $", "rank": "דירוג", "reason": "סיבה"})
    st.dataframe(dft, width='stretch', hide_index=True)
else:
    st.info("אין עסקאות עדיין.")

st.caption(f"עודכן: {datetime.now():%Y-%m-%d %H:%M} · תיק נייר · אינו ייעוץ השקעות")
