"""
ת"א 125 — סוכן איתותי מסחר ויזואלי
אסטרטגיית v9-13: No-TP in BULL + Risk-Parity + Bond rotation

הפעלה:
  python -m streamlit run signal_agent.py
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import json
import os
import sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# שעון ישראל — חשוב כי Streamlit Cloud רץ ב-UTC; בלי זה חותמת הזמן מוצגת 3 שעות אחורה.
IL_TZ = ZoneInfo("Asia/Jerusalem")

def now_il():
    return datetime.now(IL_TZ)


def is_mobile() -> bool:
    """זיהוי מכשיר נייד לפי User-Agent — לפריסה מותאמת-נייד אמיתית (לא טלאי CSS)."""
    try:
        ua = st.context.headers.get("User-Agent", "") or ""
    except Exception:
        ua = ""
    return any(k in ua for k in ("Mobile", "Android", "iPhone", "iPad", "iPod"))


def metric_grid(items, per_row_desktop, per_row_mobile=2):
    """מציג מטריקות ברשת שמסתגלת למכשיר: בנייד פחות עמודות בשורה (ברירת מחדל 2),
    בדסקטופ הפריסה הרחבה. items = רשימת dict עם label, value, ואופציונלי delta/delta_color."""
    per_row = per_row_mobile if is_mobile() else per_row_desktop
    for i in range(0, len(items), per_row):
        chunk = items[i:i + per_row]
        cols = st.columns(len(chunk))
        for col, it in zip(cols, chunk):
            with col:
                st.metric(it["label"], it["value"], it.get("delta"),
                          delta_color=it.get("delta_color", "normal"))
# הספריות הכבדות (plotly / yfinance / ta) נטענות רק אחרי שער הסיסמה — ראה main().
# כך מסך הכניסה עולה מיידית וצורך מעט זיכרון (חשוב ב-Streamlit Cloud).

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ת\"א 125 — איתותי מסחר",
    page_icon="📈",
    layout="wide",   # רחב בדסקטופ; בנייד (<730px) ממילא תופס רוחב מלא ונערם טבעית
    initial_sidebar_state="collapsed",   # מתחיל סגור; נפתח בכפתור ההמבורגר
)

# ─── Hebrew RTL style ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  body { direction: rtl; }
  .stApp { font-family: 'Segoe UI', Arial, sans-serif; }
  .regime-bull  { background:#0d6b3a; color:white; padding:12px 20px; border-radius:8px;
                  font-size:1.3rem; font-weight:bold; text-align:center; }
  .regime-neutral { background:#b87c00; color:white; padding:12px 20px; border-radius:8px;
                    font-size:1.3rem; font-weight:bold; text-align:center; }
  .regime-bear  { background:#8b1a1a; color:white; padding:12px 20px; border-radius:8px;
                  font-size:1.3rem; font-weight:bold; text-align:center; }
  .signal-buy   { background:#e6f4ea; border-left:4px solid #2d8c4e; padding:8px; border-radius:4px; }
  .signal-sell  { background:#fce8e8; border-left:4px solid #c0392b; padding:8px; border-radius:4px; }
  .signal-hold  { background:#fff8e6; border-left:4px solid #e67e22; padding:8px; border-radius:4px; }
  .metric-box   { background:#f8f9fa; border-radius:8px; padding:10px; text-align:center; }

  /* ===== התאמה לנייד — מינימלי ובטוח =====
     לא נוגעים בפריסת העמודות! Streamlit עורם אותן לבד בנייד. כאן רק התאמות
     בטוחות שלא יכולות לשבור מבנה: שוליים, גודל גופן, וגלילה לטבלאות. */
  @media (max-width: 640px) {
    .block-container { padding: 0.8rem 0.6rem !important; }
    .regime-bull, .regime-neutral, .regime-bear {
        font-size: 1rem !important; padding: 8px 10px !important;
    }
    [data-testid="stDataFrame"] { overflow-x: auto !important; }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.1rem !important; }

    /* בנייד: מסתירים לגמרי את סרגל ההגדרות (לא נחוץ, וגרם ל"מריחה" אנכית) */
    section[data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] { display: none !important; }

    /* מתג המעבר בין מסכים (radio) — שיתגלגל לשורה הבאה אם צר מדי, כך ששני
       הלחצנים תמיד גלויים גם ב-portrait (לא ייחתכו מחוץ למסך) */
    [data-testid="stRadio"] [role="radiogroup"] {
        flex-wrap: wrap !important; gap: 6px 14px !important;
    }
  }
</style>
""", unsafe_allow_html=True)


# ─── Universe ──────────────────────────────────────────────────────────────────
TA125_UNIVERSE = [
    "POLI.TA","LUMI.TA","DSCT.TA","FIBI.TA",
    "NICE.TA","CAMT.TA","TSEM.TA","NVMI.TA",
    "ESLT.TA","TEVA.TA","ICL.TA","BEZQ.TA",
    "SKBN.TA","RSEL.TA",
    "HARL.TA","MGDL.TA",
    "AZRG.TA","AMOT.TA","ALHE.TA","ELCO.TA",
    "ENLT.TA","DLEKG.TA","ILCO.TA",
]
INDEX_TICKER = "^TA125.TA"
RS_LOOKBACK  = 63

# מקור אמת יחיד לשמות/מספרי נייר/אג"ח — ראה securities.py
from securities import HEBREW_NAMES, SEC_NUMBERS, BOND_INSTRUMENTS, hname


# ─── Portfolio persistence ─────────────────────────────────────────────────────
POSITIONS_FILE = os.path.join(os.path.dirname(__file__), "positions.json")

def load_positions():
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_positions(positions):
    with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(positions, f, ensure_ascii=False, indent=2)

def analyze_position(sym, pos, df, sig, regime):
    """
    Evaluate an open position.
    Returns dict with: pnl_pct, max_since_entry, trail_stop, recommendation, sell_low, sell_high, reason
    """
    entry_price = pos["entry_price"]
    entry_date  = pd.Timestamp(pos["entry_date"])
    c = sig["price"] if sig else float(df["Close"].iloc[-1])
    atr = sig["atr"] if sig else compute_atr(df["Close"], df["High"], df["Low"])

    # Slice history since entry
    df_since = df[df.index >= entry_date]
    if df_since.empty:
        df_since = df.tail(5)

    max_high = float(df_since["High"].max()) if not df_since.empty else c

    # Trail stop: max_high - ATR*3.5, floor at entry*0.92
    trail_atr  = (max_high - atr * 3.5) if atr else entry_price * 0.85
    trail_hard = entry_price * 0.92
    trail_stop = max(trail_atr, trail_hard)

    pnl_pct = (c - entry_price) / entry_price * 100

    # Determine recommendation
    n_bear = sig["n_bear"] if sig else 0
    n_bull = sig["n_bull"] if sig else 0

    if c <= trail_stop:
        rec = "SELL_NOW"
        reason = f"Trail Stop הופעל — מחיר ({c:,.0f}) מתחת לסטופ ({trail_stop:,.0f})"
    elif regime == "BEAR":
        rec = "SELL_NOW"
        reason = "שוק דובי — האסטרטגיה דורשת יציאה מכל הפוזיציות"
    elif regime == "NEUTRAL" and pos.get("entry_regime") == "BULL" and n_bear >= 3:
        rec = "SELL"
        reason = "המשטר השתנה מ-BULL ל-NEUTRAL עם איתותים דוביים"
    elif n_bear >= 4 and n_bear > n_bull + 1:
        rec = "SELL"
        reason = f"{n_bear} איתותים דוביים — שקול מכירה"
    elif n_bear >= 2 and pnl_pct > 20:
        rec = "CAUTION"
        reason = f"רווח של {pnl_pct:.1f}% עם איתותים מעורבים — שמור על trail stop"
    else:
        rec = "HOLD"
        reason = "המגמה בעינה — המשך להחזיק עם trail stop"

    # Sell range
    if rec in ("SELL_NOW", "SELL"):
        sell_low  = c * 0.995
        sell_high = c * 1.005
    else:
        sell_low = sell_high = None

    return {
        "pnl_pct":       round(pnl_pct, 1),
        "max_since_entry": round(max_high, 0),
        "trail_stop":    round(trail_stop, 0),
        "trail_pct":     round((c - trail_stop) / c * 100, 1),
        "recommendation": rec,
        "reason":        reason,
        "sell_low":      round(sell_low,  0) if sell_low  else None,
        "sell_high":     round(sell_high, 0) if sell_high else None,
        "current_price": round(c, 0),
    }

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _last(s):
    v = s.iloc[-1] if len(s) else None
    return float(v) if v is not None and pd.notna(v) else None

def _prev(s):
    v = s.iloc[-2] if len(s) > 1 else None
    return float(v) if v is not None and pd.notna(v) else None

def compute_atr(close, high, low, window=14):
    if len(close) < window + 1: return None
    s = ta.volatility.average_true_range(high, low, close, window)
    v = s.iloc[-1]; return float(v) if pd.notna(v) else None

@st.cache_data(ttl=900)
def fetch_data():
    """Fetch all data — cached 15 minutes."""
    end   = datetime.today()
    start = end - timedelta(days=400)

    idx = yf.Ticker(INDEX_TICKER).history(start=start, end=end)
    idx.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in idx.index])

    stocks = {}
    for sym in TA125_UNIVERSE:
        try:
            df = yf.Ticker(sym).history(start=start - timedelta(days=100), end=end)
            if df.empty or len(df) < 50: continue
            df.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in df.index])
            stocks[sym] = df
        except Exception:
            pass
    return idx, stocks

def classify_regime(idx):
    if idx is None or len(idx) < 60: return "BULL", {}
    close = idx["Close"]
    high  = idx["High"]
    low   = idx["Low"]
    sma200 = _last(ta.trend.sma_indicator(close, min(200, len(close)-1)))
    sma50s = ta.trend.sma_indicator(close, min(50, len(close)-1))
    adx    = _last(ta.trend.adx(high, low, close, 14))
    c      = float(close.iloc[-1])
    sna    = sma50s.dropna()
    slope  = (float(sna.iloc[-1]) - float(sna.iloc[-31])) / float(sna.iloc[-31]) if len(sna) >= 31 else 0.0

    if sma200 is None: return "NEUTRAL", {}
    is_bull = c > sma200 and slope > 0.008 and (adx or 0) > 22
    is_bear = c < sma200 and slope < -0.008

    details = {
        "price":   round(c, 0),
        "sma200":  round(sma200, 0) if sma200 else None,
        "sma50_slope": round(slope * 100, 3),
        "adx":     round(adx, 1) if adx else None,
        "above_sma200": c > sma200 if sma200 else None,
    }
    if is_bull: return "BULL", details
    if is_bear: return "BEAR", details
    return "NEUTRAL", details

def relative_strength(stock_df, idx, lookback=RS_LOOKBACK):
    sh = stock_df["Close"]
    ih = idx["Close"]
    if len(sh) < lookback + 1 or len(ih) < lookback + 1: return None
    aligned_ih = ih[ih.index <= sh.index[-1]]
    if len(aligned_ih) < lookback + 1: return None
    return round(
        (float(sh.iloc[-1]) / float(sh.iloc[-lookback]) - 1) * 100 -
        (float(aligned_ih.iloc[-1]) / float(aligned_ih.iloc[-lookback]) - 1) * 100, 2
    )

def analyze_stock(sym, df, idx, regime, portfolio_nis):
    """Full technical analysis for a stock. Returns signal dict."""
    if df is None or len(df) < 30:
        return None

    close = df["Close"]; high = df["High"]; low = df["Low"]
    c = float(close.iloc[-1])

    # Indicators
    sma20  = ta.trend.sma_indicator(close, 20)
    sma50  = ta.trend.sma_indicator(close, 50)
    sma200 = ta.trend.sma_indicator(close, min(200, len(close)-1))
    ema12  = ta.trend.ema_indicator(close, 12)
    ema26  = ta.trend.ema_indicator(close, 26)
    rsi14  = ta.momentum.rsi(close, 14)
    macd_o = ta.trend.MACD(close, 12, 26, 9)
    macd_l = macd_o.macd(); macd_s = macd_o.macd_signal()
    stoch  = ta.momentum.StochasticOscillator(high, low, close, 14, smooth_window=3)
    bb     = ta.volatility.BollingerBands(close, 20, window_dev=2)
    atr    = compute_atr(close, high, low)

    vs20   = _last(sma20);  vs50  = _last(sma50);  vs200 = _last(sma200)
    ve12   = _last(ema12);  ve26  = _last(ema26)
    vrsi   = _last(rsi14)
    vml    = _last(macd_l); vms   = _last(macd_s)
    pvml   = _prev(macd_l); pvms  = _prev(macd_s)
    vsk    = _last(stoch.stoch()); vsd = _last(stoch.stoch_signal())
    vbbu   = _last(bb.bollinger_hband()); vbbl = _last(bb.bollinger_lband())
    vbbm   = _last(bb.bollinger_mavg())

    bb_pct = ((c - vbbl) / (vbbu - vbbl)) if vbbu and vbbl and (vbbu - vbbl) > 0 else None

    # Build bull/bear signal lists
    bull, bear = [], []

    if vrsi is not None:
        if vrsi < 30:   bull.append(f"RSI מכור-יתר ({vrsi:.0f})")
        elif vrsi < 40: bull.append(f"RSI נמוך ({vrsi:.0f})")
        elif vrsi > 70: bear.append(f"RSI קנוי-יתר ({vrsi:.0f})")
        elif vrsi > 60: bear.append(f"RSI גבוה ({vrsi:.0f})")

    if all(x is not None for x in [vml, vms, pvml, pvms]):
        if vml > vms and pvml <= pvms:   bull.append("MACD חציית מעלה")
        elif vml < vms and pvml >= pvms: bear.append("MACD חציית מטה")
        if vml > vms:  bull.append("MACD מעל קו איתות")
        else:          bear.append("MACD מתחת קו איתות")

    if vs20:
        if c > vs20:  bull.append(f"מעל SMA20 ({vs20:,.0f})")
        else:         bear.append(f"מתחת SMA20 ({vs20:,.0f})")
    if vs50:
        if c > vs50:  bull.append(f"מעל SMA50 ({vs50:,.0f})")
        else:         bear.append(f"מתחת SMA50 ({vs50:,.0f})")
    if vs200 and not pd.isna(vs200):
        if c > vs200: bull.append(f"מעל SMA200 ({vs200:,.0f})")
        else:         bear.append(f"מתחת SMA200 ({vs200:,.0f})")
    if ve12 and ve26:
        if ve12 > ve26: bull.append("EMA12 > EMA26")
        else:           bear.append("EMA12 < EMA26")

    if bb_pct is not None:
        if c < vbbl:        bull.append("מתחת לBB תחתון")
        elif c > vbbu:      bear.append("מעל לBB עליון")
        elif bb_pct < 0.25: bull.append(f"בתחתית ערוץ BB ({bb_pct*100:.0f}%)")
        elif bb_pct > 0.75: bear.append(f"בראש ערוץ BB ({bb_pct*100:.0f}%)")

    if vsk is not None:
        if vsk < 25:  bull.append(f"Stochastic מכור-יתר ({vsk:.0f})")
        elif vsk > 75: bear.append(f"Stochastic קנוי-יתר ({vsk:.0f})")

    nb, nb2 = len(bull), len(bear)

    # Relative Strength
    rs = relative_strength(df, idx) if idx is not None else None

    # Mean-reversion entry (NEUTRAL)
    is_mr_entry = (bb_pct is not None and vrsi is not None and
                   bb_pct < 0.25 and vrsi < 38)

    # Determine signal
    signal = "NEUTRAL"
    signal_strength = 0

    if regime == "BULL":
        if nb >= 4 and nb > nb2 + 1 and rs is not None and rs > 3.0:
            signal = "BUY"
            signal_strength = nb + (rs / 10)
        elif nb2 >= 4 and nb2 > nb + 1:
            signal = "SELL"
            signal_strength = -nb2
    elif regime == "NEUTRAL":
        if is_mr_entry and nb > nb2:
            signal = "BUY"
            signal_strength = nb - nb2
        elif nb2 >= 4 and nb2 > nb + 1:
            signal = "SELL"
            signal_strength = -nb2
    elif regime == "BEAR":
        if nb2 >= 2:
            signal = "SELL"
        else:
            signal = "AVOID"

    # Price levels
    stop_loss_atr  = (c - atr * 3.5)  if atr else c * 0.92
    stop_loss_hard = c * 0.92
    stop_loss      = max(stop_loss_atr, stop_loss_hard)

    tp_bull    = None  # No cap in BULL
    tp_neutral = c * 1.30 if regime == "NEUTRAL" else None

    # Entry range: current ± 0.5% (buy at market) or bb_lower–sma20 range
    entry_low  = min(c * 0.99, vbbl or c * 0.98) if signal == "BUY" else None
    entry_high = c * 1.005 if signal == "BUY" else None

    # Risk-parity position size
    stop_dist = c - stop_loss
    if stop_dist > 0 and portfolio_nis > 0:
        risk_amount  = portfolio_nis * 0.015
        qty          = risk_amount / stop_dist
        position_val = qty * c
        pos_pct      = min(position_val / portfolio_nis * 100, 20.0)
    else:
        pos_pct = 0.0

    return {
        "symbol":   sym,
        "price":    c,
        "signal":   signal,
        "strength": signal_strength,
        "bull_signals": bull,
        "bear_signals": bear,
        "n_bull":   nb,
        "n_bear":   nb2,
        "rs":       rs,
        "rsi":      vrsi,
        "bb_pct":   bb_pct,
        "atr":      atr,
        "stop_loss": stop_loss,
        "stop_pct":  ((c - stop_loss) / c * 100) if stop_loss else None,
        "tp_bull":   tp_bull,
        "tp_neutral": tp_neutral,
        "entry_low":  entry_low,
        "entry_high": entry_high,
        "pos_pct":    round(pos_pct, 1),
        "is_mr":      is_mr_entry,
        "sma20": vs20, "sma50": vs50, "sma200": vs200,
        "bb_upper": vbbu, "bb_lower": vbbl, "bb_mid": vbbm,
        "macd": vml, "macd_sig": vms,
        "df": df,  # keep for charting
    }


def make_chart(s):
    """Build Plotly candlestick chart with indicators for a stock signal."""
    df = s["df"].tail(120)  # last ~6 months

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=(hname(s["symbol"]), "RSI (14)", "MACD")
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="מחיר", increasing_fillcolor="#2d8c4e", decreasing_fillcolor="#c0392b",
        increasing_line_color="#2d8c4e", decreasing_line_color="#c0392b",
    ), row=1, col=1)

    # SMAs
    close = df["Close"]
    sma20  = ta.trend.sma_indicator(close, 20)
    sma50  = ta.trend.sma_indicator(close, 50)
    sma200 = ta.trend.sma_indicator(close, min(200, len(close)-1))

    fig.add_trace(go.Scatter(x=df.index, y=sma20,  name="SMA20",
                             line=dict(color="#3498db", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma50,  name="SMA50",
                             line=dict(color="#e67e22", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma200, name="SMA200",
                             line=dict(color="#e74c3c", width=2, dash="dot")), row=1, col=1)

    # Bollinger Bands
    bb   = ta.volatility.BollingerBands(close, 20, 2)
    fig.add_trace(go.Scatter(x=df.index, y=bb.bollinger_hband(), name="BB עליון",
                             line=dict(color="rgba(128,128,128,0.4)", width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb.bollinger_lband(), name="BB תחתון",
                             line=dict(color="rgba(128,128,128,0.4)", width=1), showlegend=False,
                             fill="tonexty", fillcolor="rgba(200,200,200,0.08)"), row=1, col=1)

    # Entry zone, stop loss, TP lines
    if s["signal"] == "BUY":
        if s["entry_low"] and s["entry_high"]:
            fig.add_hrect(y0=s["entry_low"], y1=s["entry_high"],
                          fillcolor="rgba(45,140,78,0.15)", line_width=0,
                          annotation_text="כניסה", annotation_position="right", row=1, col=1)
        if s["stop_loss"]:
            fig.add_hline(y=s["stop_loss"], line_color="#e74c3c", line_dash="dash",
                          annotation_text=f"סטופ {s['stop_loss']:,.0f}", row=1, col=1)
        if s["tp_neutral"]:
            fig.add_hline(y=s["tp_neutral"], line_color="#f39c12", line_dash="dash",
                          annotation_text=f"יעד {s['tp_neutral']:,.0f}", row=1, col=1)
    elif s["signal"] == "SELL" and s["stop_loss"]:
        fig.add_hline(y=s["stop_loss"], line_color="#e74c3c", line_dash="dash",
                      annotation_text=f"סטופ {s['stop_loss']:,.0f}", row=1, col=1)

    # RSI
    rsi = ta.momentum.rsi(close, 14)
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI",
                             line=dict(color="#9b59b6", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_color="red",   line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_color="green", line_dash="dot", row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(200,200,200,0.05)", row=2, col=1)

    # MACD
    mo = ta.trend.MACD(close, 12, 26, 9)
    macd_diff = mo.macd_diff()
    colors = ["#2d8c4e" if v >= 0 else "#c0392b" for v in macd_diff.fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=macd_diff, name="MACD Hist",
                         marker_color=colors, showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=mo.macd(),       name="MACD",
                             line=dict(color="#3498db", width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=mo.macd_signal(), name="Signal",
                             line=dict(color="#e67e22", width=1.2)), row=3, col=1)

    fig.update_layout(
        height=550, xaxis_rangeslider_visible=False,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_yaxes(gridcolor="#1e2130", row=1, col=1)
    fig.update_yaxes(gridcolor="#1e2130", row=2, col=1)
    fig.update_yaxes(gridcolor="#1e2130", row=3, col=1)
    fig.update_xaxes(gridcolor="#1e2130")
    return fig


# ─── Paper Trading DB helpers ─────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "paper_trades.db")

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@st.cache_data(ttl=120)
def load_paper_data():
    """Load all paper trading data from SQLite. Cached 2 min."""
    if not os.path.exists(DB_PATH):
        return None, None, None, None, None
    try:
        conn = _db()
        snaps  = pd.read_sql("SELECT * FROM daily_snapshots ORDER BY date", conn)
        pos    = pd.read_sql("SELECT * FROM positions WHERE status='OPEN'", conn)
        trades = pd.read_sql("SELECT * FROM trades ORDER BY ts DESC LIMIT 200", conn)
        grlog  = pd.read_sql("SELECT * FROM guardrail_log ORDER BY ts DESC LIMIT 30", conn)
        orders = pd.read_sql(
            "SELECT * FROM orders WHERE status IN ('PENDING','PARTIAL') ORDER BY created_ts DESC",
            conn)
        conn.close()
        return snaps, pos, trades, grlog, orders
    except Exception:
        return None, None, None, None, None

def paper_equity_chart(snaps, initial=100_000):
    """Equity curve + benchmark placeholder."""
    if snaps is None or snaps.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=snaps["date"], y=snaps["portfolio_value"],
        name="תיק Paper Trading",
        line=dict(color="#2d8c4e", width=2.5),
        fill="tozeroy", fillcolor="rgba(45,140,78,0.08)"
    ))
    fig.add_hline(y=initial, line_dash="dot", line_color="#888",
                  annotation_text=f"התחלה ₪{initial:,.0f}")
    fig.update_layout(
        height=280, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"), margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(gridcolor="#1e2130"),
        xaxis=dict(gridcolor="#1e2130"),
        legend=dict(orientation="h", y=1.05),
    )
    return fig

@st.cache_data(ttl=600, show_spinner=False)
def _cached_missed_days():
    """ימי מסחר שהוחמצו (נתון יקר — fetch מהמדד; נשמר במטמון ל-10 דק')."""
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from auto_trader import get_missed_trading_days
        return get_missed_trading_days()
    except Exception:
        return []


def _run_catchup_ui():
    """Run auto_trader catchup and return number of days processed."""
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from auto_trader import run_catchup, get_missed_trading_days
        missed = get_missed_trading_days()
        if not missed:
            return 0
        run_catchup(status_cb=lambda m: None)
        _cached_missed_days.clear()   # רענן את המטמון אחרי הרצה
        return len(missed)
    except Exception as e:
        st.error(f"שגיאה בהרצה: {e}")
        return 0


def paper_tab(snaps, pos_df, trades_df, grlog_df, orders_df, stocks, idx):
    """Render the full Paper Trading tab."""
    INITIAL = 100_000.0

    # ── Catchup / run-now button ──────────────────────────────────────────────
    missed = _cached_missed_days()

    needs_update = len(missed) > 0
    col_hdr, col_btn = st.columns([3, 1])
    with col_btn:
        btn_label = f"⟳ עדכן ({len(missed)} יום)" if needs_update else "⟳ הרץ עכשיו"
        if st.button(btn_label, type="primary" if needs_update else "secondary",
                     use_container_width=True):
            with st.spinner("מריץ סימולציה…"):
                n = _run_catchup_ui()
            if n:
                st.success(f"עודכן {n} ימים!")
                st.rerun()
            else:
                st.info("אין ימים שהוחמצו.")

    if snaps is None or snaps.empty:
        st.info("מנוע המסחר האוטומטי עוד לא הריץ אף יום. לחץ 'הרץ עכשיו' למעלה.")
        return

    last      = snaps.iloc[-1]
    cum_pct   = float(last["cumulative_pct"]) if last["cumulative_pct"] else 0.0
    port_val  = float(last["portfolio_value"])
    last_date = last["date"]
    regime    = last["regime"]
    n_pos     = int(last["open_positions"])
    cash_val  = float(last["cash"])
    bond_val  = float(last["bonds"])
    equity_val = max(port_val - cash_val - bond_val, 0)

    first_date = snaps.iloc[0]["date"]
    idx_now = float(idx["Close"].iloc[-1])
    try:
        idx_start = float(idx.loc[idx.index >= first_date]["Close"].iloc[0])
        idx_pct = (idx_now / idx_start - 1) * 100
    except Exception:
        idx_pct = 0.0
    alpha = cum_pct - idx_pct

    # ── Summary metrics — רשת מסתגלת למכשיר (4 בדסקטופ, 2 בשורה בנייד) ─────────
    days_stale = (now_il().date() - datetime.strptime(last_date, "%Y-%m-%d").date()).days
    stale_label = f"⚠ {days_stale} ימים ישן" if days_stale > 1 else f"Regime: {regime}"
    metric_grid([
        {"label": "שווי תיק", "value": f"₪{port_val:,.0f}", "delta": f"{cum_pct:+.2f}%"},
        {"label": "תשואה מצטברת", "value": f"{cum_pct:+.2f}%"},
        {"label": "ת\"א 125 (אותה תקופה)", "value": f"{idx_pct:+.2f}%"},
        {"label": "אלפא vs מדד", "value": f"{alpha:+.2f}%",
         "delta_color": "normal" if alpha >= 0 else "inverse"},
        {"label": "מניות", "value": f"₪{equity_val:,.0f}", "delta": f"{equity_val/port_val*100:.0f}% מהתיק"},
        {"label": "אגח", "value": f"₪{bond_val:,.0f}", "delta": f"{bond_val/port_val*100:.0f}% מהתיק"},
        {"label": "מזומן", "value": f"₪{cash_val:,.0f}", "delta": f"{cash_val/port_val*100:.0f}% מהתיק"},
        {"label": "הרצה אחרונה", "value": last_date, "delta": stale_label,
         "delta_color": "inverse" if days_stale > 1 else "normal"},
    ], per_row_desktop=4, per_row_mobile=2)

    st.markdown("")

    # ── Equity curve ──────────────────────────────────────────────────────────
    fig = paper_equity_chart(snaps, INITIAL)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Full holdings table ───────────────────────────────────────────────────
    st.markdown("**פירוט החזקות מלא**")
    holdings = []

    # מניות
    if pos_df is not None and not pos_df.empty:
        for _, p in pos_df.iterrows():
            sym = p["symbol"]
            ep  = float(p["entry_price"])
            qty = float(p["qty"])
            cur = ep
            if sym in stocks:
                try: cur = float(stocks[sym]["Close"].iloc[-1])
                except: pass
            atr = None
            if sym in stocks:
                df_s = stocks[sym]
                try:
                    atr = float(ta.volatility.average_true_range(
                        df_s["High"], df_s["Low"], df_s["Close"], 14
                    ).dropna().iloc[-1])
                except: pass
            trail_high = float(p["trail_high"])
            trail_stop = max(
                trail_high - atr * 3.5 if atr else ep * 0.85,
                ep * 0.92
            )
            val = qty * cur
            pnl = (cur / ep - 1) * 100
            holdings.append({
                "סוג":        "מניה",
                "נייר":       hname(sym),
                "מחיר כניסה": f"₪{ep:,.0f}",
                "מחיר נוכחי": f"₪{cur:,.0f}",
                "כמות":       f"{qty:,.0f}",
                "שווי":        f"₪{val:,.0f}",
                "% מהתיק":    f"{val/port_val*100:.1f}%",
                "רווח/הפסד":  f"{pnl:+.1f}%",
                "Trail Stop": f"₪{trail_stop:,.0f}",
                "ימים":       str(int(p["days_held"])),
            })

    # אגח
    if bond_val > 0:
        bond_instruments = BOND_INSTRUMENTS.get(regime, BOND_INSTRUMENTS["NEUTRAL"])
        if bond_instruments:
            split = bond_val / len(bond_instruments)
            for name, secn in bond_instruments:
                holdings.append({
                    "סוג":        "אגח",
                    "נייר":       f"{name} · נ\"ע {secn}",
                    "מחיר כניסה": "ממוצע שוק",
                    "מחיר נוכחי": "~3.8% שנתי",
                    "כמות":       "—",
                    "שווי":        f"₪{split:,.0f}",
                    "% מהתיק":    f"{split/port_val*100:.1f}%",
                    "רווח/הפסד":  "+3.8% שנתי",
                    "Trail Stop": "—",
                    "ימים":       "—",
                })
        else:
            holdings.append({
                "סוג": "אגח", "נייר": "אגח ממשלתי (BULL — 0%)",
                "מחיר כניסה": "—", "מחיר נוכחי": "—", "כמות": "—",
                "שווי": "₪0", "% מהתיק": "0%",
                "רווח/הפסד": "—", "Trail Stop": "—", "ימים": "—",
            })

    # מזומן
    holdings.append({
        "סוג":        "מזומן",
        "נייר":       "מזומן / עו\"ש",
        "מחיר כניסה": "—",
        "מחיר נוכחי": "—",
        "כמות":       "—",
        "שווי":        f"₪{cash_val:,.0f}",
        "% מהתיק":    f"{cash_val/port_val*100:.1f}%",
        "רווח/הפסד":  "—",
        "Trail Stop": "—",
        "ימים":       "—",
    })

    df_hold = pd.DataFrame(holdings)
    styler_h = df_hold.style
    sf_h = getattr(styler_h, "map", getattr(styler_h, "applymap", None))
    def color_type(v):
        if v == "מניה":  return "color:#2d8c4e;font-weight:bold"
        if v == "אגח":   return "color:#3498db;font-weight:bold"
        if v == "מזומן": return "color:#e67e22"
        return ""
    def color_pnl2(v):
        if isinstance(v, str) and "%" in v and "שנתי" not in v:
            try:
                n = float(v.replace("%","").replace("+",""))
                return "color:#2d8c4e;font-weight:bold" if n > 0 else ("color:#c0392b" if n < 0 else "")
            except: pass
        return ""
    st.dataframe(
        sf_h(color_type, subset=["סוג"]),
        use_container_width=True, hide_index=True,
        height=min(40 + len(holdings) * 36, 500)
    )

    # ── Pending orders (ladder) ───────────────────────────────────────────────
    if orders_df is not None and not orders_df.empty:
        st.markdown("---")
        st.markdown(f"**פקודות ממתינות — Ladder Entry ({len(orders_df)})**")
        disp_o = orders_df[["created_ts","symbol","order_type","limit_price",
                             "total_qty","attempts","max_attempts","tranche","status"]].copy()
        disp_o["created_ts"]   = disp_o["created_ts"].str[:10]
        disp_o["symbol"]       = disp_o["symbol"].apply(hname)
        disp_o["limit_price"]  = disp_o["limit_price"].apply(lambda x: f"₪{x:,.0f}")
        disp_o["total_qty"]    = disp_o["total_qty"].apply(lambda x: f"{x:,.0f}")
        disp_o["ניסיון"]       = disp_o.apply(
            lambda r: f"{int(r['attempts'])}/{int(r['max_attempts'])}", axis=1)
        disp_o["טרנש"]         = disp_o["tranche"].apply(lambda x: f"T{int(x)}")
        disp_o = disp_o[["created_ts","symbol","order_type","limit_price",
                          "total_qty","טרנש","ניסיון","status"]]
        disp_o.columns = ["תאריך","מניה","סוג","מחיר לימיט","כמות","טרנש","ניסיון","סטטוס"]
        styler_o = disp_o.style
        sf_o = getattr(styler_o, "map", getattr(styler_o, "applymap", None))
        def color_order_type(v):
            if v == "MARKET": return "color:#f39c12;font-weight:bold"
            if v == "LIMIT":  return "color:#3498db"
            return ""
        st.dataframe(sf_o(color_order_type, subset=["סוג"]),
                     use_container_width=True, hide_index=True)
        st.caption("T1=מחיר שוק (50%) | T2=1% מתחת (30%) | T3=2% מתחת (20%) | מבוטל אחרי 3 ימים ללא מילוי")

    st.markdown("---")
    col_l, col_r = st.columns([1, 1])

    # ── Pie breakdown ─────────────────────────────────────────────────────────
    with col_l:
        st.markdown("**הרכב התיק**")
        pie_labels = ["מניות", "אגח", "מזומן"]
        pie_values = [equity_val, bond_val, cash_val]
        pie_colors = ["#2d8c4e", "#3498db", "#e67e22"]
        pie = go.Figure(go.Pie(
            labels=pie_labels, values=pie_values,
            marker_colors=pie_colors, hole=0.5,
            textinfo="label+percent+value",
            texttemplate="%{label}<br>%{percent}<br>₪%{value:,.0f}",
        ))
        pie.update_layout(
            height=300, paper_bgcolor="#0e1117",
            font=dict(color="white"), showlegend=False,
            margin=dict(l=0, r=0, t=10, b=10),
        )
        st.plotly_chart(pie, use_container_width=True)

    # ── Trade history ─────────────────────────────────────────────────────────
    with col_r:
        st.markdown("**היסטוריית עסקאות**")
        if trades_df is None or trades_df.empty:
            st.write("אין עסקאות עדיין.")
        else:
            disp = trades_df[["ts","symbol","action","price","pnl_pct","reason"]].copy()
            disp["ts"]      = disp["ts"].str[:10]
            disp["symbol"]  = disp["symbol"].apply(hname)
            disp["price"]   = disp["price"].apply(lambda x: f"₪{x:,.0f}")
            disp["pnl_pct"] = disp["pnl_pct"].apply(
                lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
            disp.columns = ["תאריך","מניה","פעולה","מחיר","רווח/הפסד","סיבה"]
            styler2 = disp.style
            sf2 = getattr(styler2, "map", getattr(styler2, "applymap", None))
            def color_action(v):
                if v == "BUY":  return "color:#2d8c4e;font-weight:bold"
                if v == "SELL": return "color:#c0392b;font-weight:bold"
                return ""
            st.dataframe(sf2(color_action, subset=["פעולה"]),
                         use_container_width=True, hide_index=True, height=300)

    # ── Guardrail log ─────────────────────────────────────────────────────────
    if grlog_df is not None and not grlog_df.empty:
        st.markdown("---")
        with st.expander("יומן Guardrails"):
            disp_g = grlog_df[["ts","rule","detail","blocked"]].copy()
            disp_g["ts"] = disp_g["ts"].str[:16]
            disp_g["blocked"] = disp_g["blocked"].apply(lambda x: "חסום" if x else "אזהרה")
            disp_g.columns = ["זמן","חוק","פרטים","סטטוס"]
            st.dataframe(disp_g, use_container_width=True, hide_index=True)


# ─── Main app ─────────────────────────────────────────────────────────────────
import hashlib as _hashlib

# SHA-256 של סיסמת הדשבורד (הסיסמה עצמה לעולם לא נשמרת בקוד/בגיט).
# ניתן לעקוף עם st.secrets["dashboard_password"] (סיסמה רגילה) אם רוצים.
_DASH_PW_HASH = "3090009d6533b344b3a4aae98c2133d3f394148243d9417f9f83dfd20d450182"


def _check_password() -> bool:
    """מציג שער סיסמה. מחזיר True רק לאחר הזנת הסיסמה הנכונה."""
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


def main():
    _check_password()
    # טעינת הספריות הכבדות רק אחרי אימות (מאיץ את מסך הסיסמה ומפחית זיכרון עד כניסה)
    global go, make_subplots, yf, ta
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import yfinance as yf
    import ta
    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.title("⚙️ הגדרות")
    portfolio_nis = st.sidebar.number_input(
        "גודל תיק (₪)", min_value=10_000, max_value=10_000_000,
        value=100_000, step=10_000, format="%d"
    )
    min_bull_signals = 4
    show_all = st.sidebar.checkbox("הצג גם NEUTRAL", value=False)

    if st.sidebar.button("🔄 רענן נתונים"):
        st.cache_data.clear()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**אסטרטגיה:** v9-13 — No-TP in BULL + Risk-Parity")
    st.sidebar.markdown("**עדכון:** כל 15 דקות")

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("טוען נתונים..."):
        idx, stocks = fetch_data()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    # מעבר בין מסכים — שני כפתורים בעמודות: נערמים אנכית בנייד (תמיד גלויים),
    # זה לצד זה בדסקטוף. הכפתור הפעיל מודגש (primary).
    if "view" not in st.session_state:
        st.session_state["view"] = "signals"
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("📡 איתותים חיים", use_container_width=True,
                     type="primary" if st.session_state["view"] == "signals" else "secondary"):
            st.session_state["view"] = "signals"
            st.rerun()
    with nav2:
        if st.button("🤖 תיק אוטומטי", use_container_width=True,
                     type="primary" if st.session_state["view"] == "portfolio" else "secondary"):
            st.session_state["view"] = "portfolio"
            st.rerun()

    if st.session_state["view"] == "signals":
        _render_signals_tab(idx, stocks, portfolio_nis, show_all)
    else:
        snaps, pos_df, trades_df, grlog_df, orders_df = load_paper_data()
        paper_tab(snaps, pos_df, trades_df, grlog_df, orders_df, stocks, idx)


def _render_signals_tab(idx, stocks, portfolio_nis, show_all):
    # ── Regime banner ─────────────────────────────────────────────────────────
    regime, regime_info = classify_regime(idx)

    REGIME_HE = {"BULL": "📈 שוק שורי — BULL", "NEUTRAL": "⚖️ שוק ניטרלי — NEUTRAL", "BEAR": "📉 שוק דובי — BEAR"}
    REGIME_CSS = {"BULL": "regime-bull", "NEUTRAL": "regime-neutral", "BEAR": "regime-bear"}
    REGIME_DESC = {
        "BULL":    "קנייה: מניות עם מומנטום + RS>3%. TP: ללא תקרה (trail stop בלבד). בונד: 0%",
        "NEUTRAL": "קנייה: היפוך מגמה (RSI<38 + BB<25%). TP: 30%. בונד: 40% מהתיק",
        "BEAR":    "אין קניות מניות. הון מופנה לאגרות חוב (60%). ממתינים לחזרת שוק שורי",
    }

    st.markdown(f'<div class="{REGIME_CSS[regime]}">{REGIME_HE[regime]}</div>', unsafe_allow_html=True)
    st.markdown(f"<small style='color:#aaa;'>**אסטרטגיה:** {REGIME_DESC[regime]}</small>", unsafe_allow_html=True)

    # Regime metrics — רשת שמסתגלת למכשיר (5 בדסקטופ, 2 בשורה בנייד)
    metric_grid([
        {"label": "מדד ת\"א 125", "value": f"₪{regime_info.get('price', 0):,.0f}"},
        {"label": "SMA200", "value": f"₪{regime_info.get('sma200', 0):,.0f}",
         "delta": "מעל ✓" if regime_info.get("above_sma200") else "מתחת ✗"},
        {"label": "שיפוע SMA50 (30י')", "value": f"{regime_info.get('sma50_slope', 0):+.3f}%"},
        {"label": "ADX", "value": f"{regime_info.get('adx', 0):.1f}"},
        {"label": "עדכון אחרון", "value": now_il().strftime("%H:%M:%S")},
    ], per_row_desktop=5, per_row_mobile=2)

    st.markdown("---")

    # ── Analyze all stocks ────────────────────────────────────────────────────
    signals = []
    for sym, df in stocks.items():
        result = analyze_stock(sym, df, idx, regime, portfolio_nis)
        if result:
            signals.append(result)

    buy_signals  = sorted([s for s in signals if s["signal"] == "BUY"],
                          key=lambda x: x["strength"], reverse=True)
    sell_signals = sorted([s for s in signals if s["signal"] == "SELL"],
                          key=lambda x: x["strength"])
    hold_signals = [s for s in signals if s["signal"] == "NEUTRAL"]

    # ── PORTFOLIO TRACKER ─────────────────────────────────────────────────────
    positions = load_positions()
    sig_map   = {s["symbol"]: s for s in signals}

    if positions:
        st.subheader(f"💼 תיק פעיל ({len(positions)} פוזיציות)")
        REC_COLOR = {
            "HOLD":     ("#e6f4ea", "#2d8c4e", "החזק"),
            "CAUTION":  ("#fff8e6", "#b87c00", "זהירות"),
            "SELL":     ("#fce8e8", "#c0392b", "מכור"),
            "SELL_NOW": ("#8b1a1a", "white",   "מכור עכשיו"),
        }

        for sym, pos in list(positions.items()):
            df_sym  = stocks.get(sym)
            sig     = sig_map.get(sym)
            if df_sym is None:
                continue
            pa = analyze_position(sym, pos, df_sym, sig, regime)
            bg, fg, rec_he = REC_COLOR.get(pa["recommendation"], ("#f0f0f0","#333","?"))
            pnl_color = "#2d8c4e" if pa["pnl_pct"] >= 0 else "#c0392b"
            pnl_sign  = "+" if pa["pnl_pct"] >= 0 else ""

            header = (
                f"**{hname(sym)}**  |  "
                f"כניסה: ₪{pos['entry_price']:,.0f} ({pos['entry_date']})  |  "
                f"כעת: ₪{pa['current_price']:,.0f}  |  "
                f"רווח/הפסד: {pnl_sign}{pa['pnl_pct']:.1f}%  |  "
                f"המלצה: {rec_he}"
            )
            with st.expander(header, expanded=(pa["recommendation"] in ("SELL_NOW","SELL"))):
                st.markdown(
                    f'<div style="background:{bg};color:{fg};padding:10px 16px;'
                    f'border-radius:6px;font-weight:bold;font-size:1.05rem;">'
                    f'{rec_he} — {pa["reason"]}</div>',
                    unsafe_allow_html=True
                )
                st.markdown("")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**מחירים**")
                    st.markdown(f"""
| פרמטר | ערך |
|-------|-----|
| מחיר כניסה | ₪{pos['entry_price']:,.0f} |
| מחיר נוכחי | ₪{pa['current_price']:,.0f} |
| שיא מאז כניסה | ₪{pa['max_since_entry']:,.0f} |
| Trail Stop | ₪{pa['trail_stop']:,.0f} ({pa['trail_pct']:.1f}% מתחת) |
""")
                with c2:
                    st.markdown("**רווח / הפסד**")
                    invested = portfolio_nis * pos.get("pos_pct", 10) / 100
                    gain_nis = invested * pa["pnl_pct"] / 100
                    st.markdown(f"""
| פרמטר | ערך |
|-------|-----|
| % שינוי | {pnl_sign}{pa['pnl_pct']:.1f}% |
| רווח/הפסד בש"ח | ₪{gain_nis:+,.0f} |
| שווי פוזיציה כיום | ₪{invested + gain_nis:,.0f} |
| מס רווחי הון (25%) | ₪{max(gain_nis*0.25,0):,.0f} |
""")
                with c3:
                    if pa["recommendation"] in ("SELL_NOW", "SELL"):
                        st.markdown("**טווח מכירה מומלץ**")
                        st.markdown(f"""
| פרמטר | ערך |
|-------|-----|
| מכור בטווח | ₪{pa['sell_low']:,.0f} – ₪{pa['sell_high']:,.0f} |
| מחיר שוק נוכחי | ₪{pa['current_price']:,.0f} |
""")
                        st.error("מכור לפי מחיר שוק — אל תחכה לטווח מדויק")
                    else:
                        st.markdown("**פרמטרי החזקה**")
                        st.markdown(f"""
| פרמטר | ערך |
|-------|-----|
| Trail Stop פעיל | ₪{pa['trail_stop']:,.0f} |
| מכור רק אם יורד ל | ₪{pa['trail_stop']:,.0f} |
| איתות נוכחי | {sig['signal'] if sig else 'N/A'} |
""")

                # כפתור הסרה מהתיק
                if st.button(f"הסר {sym} מהתיק", key=f"remove_{sym}"):
                    del positions[sym]
                    save_positions(positions)
                    st.rerun()

        st.markdown("---")

    # ── BOND recommendation ───────────────────────────────────────────────────
    if regime in ("NEUTRAL", "BEAR"):
        bond_alloc = {"NEUTRAL": 40, "BEAR": 60}[regime]
        bond_val   = portfolio_nis * bond_alloc / 100
        st.info(
            f"💰 **המלצת אג\"ח:** הפנה {bond_alloc}% מהתיק (₪{bond_val:,.0f}) "
            f"לאגרות חוב ממשלתיות ישראליות (TLBO.TA / קרן אג\"ח ממ' קצרה) — "
            f"תשואה שנתית ~3.8%. מסייע לניצול ההון בתקופות חלשות."
        )

    # ── BUY SIGNALS ──────────────────────────────────────────────────────────
    st.subheader(f"🟢 איתותי קנייה ({len(buy_signals)})")

    if not buy_signals:
        st.write("אין איתותי קנייה כרגע בהתאם למשטר השוק הנוכחי.")
    else:
        for s in buy_signals:
            entry_type = "היפוך מגמה (MR)" if s["is_mr"] else f"מומנטום (RS: {s['rs']:+.1f}%)"
            with st.expander(
                f"**{hname(s['symbol'])}**  |  ₪{s['price']:,.0f}  |  "
                f"🟢 {s['n_bull']} איתותים שוריים / {s['n_bear']} דוביים  |  "
                f"{entry_type}  |  % תיק: {s['pos_pct']:.1f}%",
                expanded=(len(buy_signals) <= 3)
            ):
                col_a, col_b, col_c = st.columns([1, 1, 2])

                with col_a:
                    st.markdown("**💰 רמות כניסה ויציאה**")
                    st.markdown(f"""
                    | פרמטר | ערך |
                    |-------|-----|
                    | מחיר נוכחי | ₪{s['price']:,.0f} |
                    | טווח כניסה | ₪{s['entry_low']:,.0f} – ₪{s['entry_high']:,.0f} |
                    | סטופ לוס (ATR) | ₪{s['stop_loss']:,.0f} ({s['stop_pct']:.1f}% מהכניסה) |
                    | יעד רווח | {"ללא תקרה (trail stop)" if not s['tp_neutral'] else f"₪{s['tp_neutral']:,.0f} (+30%)"} |
                    | ATR (14) | {f"₪{s['atr']:,.0f}" if s['atr'] else "N/A"} |
                    """)

                with col_b:
                    st.markdown("**📊 גודל פוזיציה (Risk-Parity)**")
                    risk_nis  = portfolio_nis * 0.015
                    pos_nis   = portfolio_nis * s["pos_pct"] / 100
                    st.markdown(f"""
                    | פרמטר | ערך |
                    |-------|-----|
                    | סיכון לעסקה (1.5%) | ₪{risk_nis:,.0f} |
                    | גודל פוזיציה | {s['pos_pct']:.1f}% מהתיק |
                    | שווי פוזיציה | ₪{pos_nis:,.0f} |
                    | RSI נוכחי | {s['rsi']:.0f} if s['rsi'] else "N/A" |
                    | BB Position | {f"{s['bb_pct']*100:.0f}%" if s['bb_pct'] is not None else "N/A"} |
                    """)

                with col_c:
                    st.markdown("**📋 אינדיקטורים תומכים**")
                    bull_html = "".join([f'<span style="background:#e6f4ea;padding:3px 8px;border-radius:12px;margin:2px;font-size:0.85rem;">✅ {b}</span>' for b in s["bull_signals"]])
                    bear_html = "".join([f'<span style="background:#fce8e8;padding:3px 8px;border-radius:12px;margin:2px;font-size:0.85rem;">🔴 {b}</span>' for b in s["bear_signals"]])
                    st.markdown(bull_html, unsafe_allow_html=True)
                    if s["bear_signals"]:
                        st.markdown("<small style='color:#aaa;'>אינדיקטורים נגד:</small>", unsafe_allow_html=True)
                        st.markdown(bear_html, unsafe_allow_html=True)

                    if s["rs"] is not None:
                        rs_color = "#2d8c4e" if s["rs"] > 0 else "#c0392b"
                        st.markdown(f"**חוזק יחסי vs ת\"א 125 (63י'):** "
                                    f'<span style="color:{rs_color};font-weight:bold">{s["rs"]:+.1f}%</span>',
                                    unsafe_allow_html=True)

                # Chart
                fig = make_chart(s)
                st.plotly_chart(fig, use_container_width=True)

                # Add to portfolio
                st.markdown("---")
                col_add1, col_add2, col_add3 = st.columns([1, 1, 2])
                with col_add1:
                    entry_p = st.number_input(
                        "מחיר כניסה (₪)", value=float(round(s["price"], 0)),
                        step=1.0, key=f"ep_{s['symbol']}"
                    )
                with col_add2:
                    entry_d = st.date_input(
                        "תאריך כניסה", value=datetime.today().date(),
                        format="DD/MM/YYYY",
                        key=f"ed_{s['symbol']}"
                    )
                with col_add3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button(f"➕ הוסף {s['symbol']} לתיק הפעיל", key=f"add_{s['symbol']}"):
                        positions = load_positions()
                        positions[s["symbol"]] = {
                            "entry_price":  entry_p,
                            "entry_date":   str(entry_d),
                            "pos_pct":      s["pos_pct"],
                            "entry_regime": regime,
                        }
                        save_positions(positions)
                        st.success(f"{hname(s['symbol'])} נוסף לתיק!")
                        st.rerun()

    st.markdown("---")

    # ── SELL SIGNALS ─────────────────────────────────────────────────────────
    st.subheader(f"🔴 איתותי מכירה / זהירות ({len(sell_signals)})")

    if not sell_signals:
        st.write("אין איתותי מכירה פעילים כרגע.")
    else:
        for s in sell_signals:
            with st.expander(
                f"**{hname(s['symbol'])}**  |  ₪{s['price']:,.0f}  |  "
                f"🔴 {s['n_bear']} איתותים דוביים / {s['n_bull']} שוריים"
            ):
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.markdown("**⚠️ רמות מכירה מומלצות**")
                    st.markdown(f"""
                    | פרמטר | ערך |
                    |-------|-----|
                    | מחיר נוכחי | ₪{s['price']:,.0f} |
                    | מכירה בהפסד (סטופ) | ₪{s['stop_loss']:,.0f} ({s['stop_pct']:.1f}% מתחת) |
                    | RSI | {f"{s['rsi']:.0f}" if s['rsi'] else "N/A"} |
                    | BB Position | {f"{s['bb_pct']*100:.0f}%" if s['bb_pct'] is not None else "N/A"} |
                    """)
                    st.warning(f"⚠️ מניה זו מציגה {s['n_bear']} איתותים דוביים. "
                               f"{'שקול מכירה.' if regime != 'BULL' else 'במשטר שורי — בדוק trail stop.'}")

                with col_b:
                    st.markdown("**📋 אינדיקטורים דוביים**")
                    bear_html = "".join([f'<span style="background:#fce8e8;padding:3px 8px;border-radius:12px;margin:2px;font-size:0.85rem;">🔴 {b}</span>' for b in s["bear_signals"]])
                    bull_html = "".join([f'<span style="background:#e6f4ea;padding:3px 8px;border-radius:12px;margin:2px;font-size:0.85rem;">✅ {b}</span>' for b in s["bull_signals"]])
                    st.markdown(bear_html, unsafe_allow_html=True)
                    if s["bull_signals"]:
                        st.markdown("<small style='color:#aaa;'>אינדיקטורים נגד:</small>", unsafe_allow_html=True)
                        st.markdown(bull_html, unsafe_allow_html=True)

                fig = make_chart(s)
                st.plotly_chart(fig, use_container_width=True)

    # ── ALL STOCKS TABLE ──────────────────────────────────────────────────────
    if show_all:
        st.markdown("---")
        st.subheader("📊 סקירת כל המניות")
        rows = []
        for s in sorted(signals, key=lambda x: x["strength"], reverse=True):
            rows.append({
                "מניה":        hname(s["symbol"]),
                "מחיר":        f"₪{s['price']:,.0f}",
                "איתות":       s["signal"],
                "שורי":        s["n_bull"],
                "דובי":        s["n_bear"],
                "RSI":         f"{s['rsi']:.0f}" if s["rsi"] else "-",
                "RS 63י'":    f"{s['rs']:+.1f}%" if s["rs"] is not None else "-",
                "סטופ":        f"₪{s['stop_loss']:,.0f}" if s["stop_loss"] else "-",
                "% תיק":       f"{s['pos_pct']:.1f}%" if s["signal"] == "BUY" else "-",
            })
        df_table = pd.DataFrame(rows)

        def color_signal(val):
            if val == "BUY":    return "background-color: #e6f4ea; color: #2d8c4e; font-weight:bold"
            if val == "SELL":   return "background-color: #fce8e8; color: #c0392b; font-weight:bold"
            if val == "AVOID":  return "background-color: #f0f0f0; color: #666"
            return ""

        styler = df_table.style
        style_fn = getattr(styler, "map", getattr(styler, "applymap", None))
        st.dataframe(style_fn(color_signal, subset=["איתות"]),
                     use_container_width=True, hide_index=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<small style='color:#666;'>⚠️ **אזהרה:** הכלי הזה מיועד למחקר ולימוד בלבד. "
        "האיתותים מבוססים על ניתוח טכני היסטורי ואינם מהווים ייעוץ השקעות. "
        "כל החלטת השקעה היא באחריות המשתמש.</small>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
