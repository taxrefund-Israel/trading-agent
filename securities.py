# -*- coding: utf-8 -*-
"""
מקור אמת יחיד לניירות הערך והאג"ח במערכת.
כל מספרי הנייר אומתו מול ביזפורטל / אתר הבורסה (יוני 2026).

גם הדשבורד (signal_agent) וגם שולח המיילים (notifier) מייבאים מכאן —
כך לא ייתכן שם/מספר נייר שגוי או נייר שאינו קיים בשני מקומות שונים.
"""
from __future__ import annotations

# ─── מניות ת"א 125 שבמעקב ──────────────────────────────────────────────────────
HEBREW_NAMES = {
    "POLI.TA":  "בנק הפועלים",
    "LUMI.TA":  "בנק לאומי",
    "DSCT.TA":  "בנק דיסקונט",
    "FIBI.TA":  "הבנק הבינלאומי",
    "NICE.TA":  "נייס מערכות",
    "CAMT.TA":  "קמטק",
    "TSEM.TA":  "טאואר סמיקונדקטור",
    "NVMI.TA":  "נובה",
    "ESLT.TA":  "אלביט מערכות",
    "TEVA.TA":  "טבע",
    "ICL.TA":   "ICL קבוצה",
    "BEZQ.TA":  "בזק",
    "SKBN.TA":  "שיכון ובינוי",
    "RSEL.TA":  "אר.אס.אל אלקטרוניקה",
    "HARL.TA":  "הראל השקעות",
    "MGDL.TA":  "מגדל ביטוח",
    "AZRG.TA":  "קבוצת עזריאלי",
    "AMOT.TA":  "אמות השקעות",
    "ALHE.TA":  "אלוני חץ",
    "ELCO.TA":  "אלקו",
    "ENLT.TA":  "אנלייט אנרגיה",
    "DLEKG.TA": "דלק קבוצה",
    "ILCO.TA":  "ישראל קנדה",
}

# מספר נייר בבורסה לניירות ערך בתל אביב
SEC_NUMBERS = {
    "POLI.TA":  "662577",
    "LUMI.TA":  "604611",
    "DSCT.TA":  "691212",
    "FIBI.TA":  "593038",
    "NICE.TA":  "273011",
    "CAMT.TA":  "1095264",
    "TSEM.TA":  "1082379",
    "NVMI.TA":  "1084557",
    "ESLT.TA":  "1081124",
    "TEVA.TA":  "629014",
    "ICL.TA":   "281014",
    "BEZQ.TA":  "230011",
    "SKBN.TA":  "1081942",
    "RSEL.TA":  "299016",
    "HARL.TA":  "585018",
    "MGDL.TA":  "1081165",
    "AZRG.TA":  "1119478",
    "AMOT.TA":  "1097278",
    "ALHE.TA":  "390013",
    "ELCO.TA":  "694034",
    "ENLT.TA":  "720011",
    "DLEKG.TA": "1084128",
    "ILCO.TA":  "434019",
}

# ─── הדלי ההגנתי: קרן כספית שקלית (החליפה את האג"ח) ─────────────────────────────
# קרן כספית שקלית עדיפה כאן: NAV יציב (כמעט אפס סיכון מחיר/מח"מ), עוקבת אחרי ריבית
# קצרה, ללא עמלות קנייה/מכירה, ומס 25% רק על הרווח הריאלי. נבחרה קרן גדולה, נזילה,
# בדמי ניהול נמוכים מאוד, המחזיקה רק ממשלתי/מק"ם (ללא קונצרני) — אומת מול funder.
# פורמט: (שם, מספר נייר)
MONEY_FUND = ("מגדל כספית שקלית ללא קונצרני (דמי ניהול 0.06%)", "5140785")

# אותה קרן כספית בכל מצב שבו יש הקצאה הגנתית (BULL = 0%).
BOND_INSTRUMENTS = {
    "BULL":    [],
    "NEUTRAL": [MONEY_FUND],
    "BEAR":    [MONEY_FUND],
}


def hname(sym: str) -> str:
    """מחזיר 'שם · נ"ע מספר' או הטיקר אם אין מיפוי."""
    name = HEBREW_NAMES.get(sym)
    secn = SEC_NUMBERS.get(sym)
    if name and secn:
        return f'{name} · נ"ע {secn}'
    if name:
        return f"{name} ({sym})"
    return sym


def sec_number(sym: str) -> str | None:
    return SEC_NUMBERS.get(sym)


# כתובת הדשבורד הציבורי ב-Streamlit — ברירת מחדל קבועה כשאין משתנה סביבה
# (הרצות מקומיות לא מגדירות DASHBOARD_URL, ואז נשלח קישור לוקאלהוסט חסר תועלת).
DEFAULT_DASHBOARD_URL = "https://trading-agent-3xxzhldft5fbnuy6kvpipt.streamlit.app"


def dashboard_url() -> str | None:
    """כתובת הדשבורד לקישור בהודעות. קודם משתנה הסביבה DASHBOARD_URL
    (מוגדר ב-GitHub Actions), אחרת ברירת המחדל הציבורית ב-Streamlit."""
    import os
    url = (os.environ.get("DASHBOARD_URL") or "").strip()
    return url or DEFAULT_DASHBOARD_URL
