/**
 * כלב שמירה למערכות המסחר — רץ ב-Cloudflare, בלתי תלוי ב-GitHub ובמחשב הביתי.
 * כל ערב 19:30 UTC (22:30 IL): מוודא שהריצות המתוזמנות של היום באמת קרו —
 * הסוכן הישראלי (יומי) והאיתות האמריקאי (ימי שני, כולל בדיקת קובץ הסטייט).
 * שקט כשהכול תקין; התראת טלגרם רק כשמשהו לא רץ.
 * נולד אחרי 5.10.2026 — תקלת runners של GitHub שהפילה את כל הריצות בשקט מוחלט.
 */
const REPO = "taxrefund-Israel/trading-agent";

async function gh(path) {
  const r = await fetch(`https://api.github.com/repos/${REPO}/${path}`, {
    headers: {
      "User-Agent": "trading-watchdog",
      "Accept": "application/vnd.github+json",
    },
  });
  if (!r.ok) throw new Error(`GitHub API ${r.status} על ${path}`);
  return r.json();
}

function todayUTC() {
  return new Date().toISOString().slice(0, 10);
}

async function runsToday(workflow) {
  const data = await gh(`actions/workflows/${workflow}/runs?per_page=10`);
  const today = todayUTC();
  return (data.workflow_runs || []).filter(
    (r) => r.created_at.slice(0, 10) === today
  );
}

export default {
  async scheduled(event, env, ctx) {
    const problems = [];
    const today = todayUTC();

    // ── הסוכן הישראלי: אמור לרוץ כל יום (גם no-op בימים ללא מסחר) ──
    try {
      const il = await runsToday("daily-signals.yml");
      if (il.length === 0) {
        problems.push(
          "הסוכן הישראלי: אף ריצה לא קרתה היום — המתזמן של GitHub כנראה תקוע"
        );
      } else if (!il.some((r) => r.conclusion === "success" || r.status === "in_progress" || r.status === "queued")) {
        problems.push("הסוכן הישראלי: כל הריצות של היום נכשלו");
      }
    } catch (e) {
      problems.push(`שגיאה בבדיקת הסוכן הישראלי: ${e.message}`);
    }

    // ── האיתות האמריקאי: בימי שני בלבד ──
    if (new Date().getUTCDay() === 1) {
      try {
        const us = await runsToday("us-weekly-signals.yml");
        if (!us.some((r) => r.conclusion === "success")) {
          problems.push("האיתות האמריקאי: אין ריצה מוצלחת היום");
        }
        const st = await fetch(
          `https://raw.githubusercontent.com/${REPO}/main/us_portfolio_state.json`,
          { headers: { "User-Agent": "trading-watchdog" } }
        ).then((r) => r.json());
        if (!(st.history || []).some((h) => h.date === today)) {
          problems.push(
            "האיתות האמריקאי: אין שורת איתות של היום בקובץ הסטייט"
          );
        }
      } catch (e) {
        problems.push(`שגיאה בבדיקת האיתות האמריקאי: ${e.message}`);
      }
    }

    if (problems.length) {
      const text =
        "🐶 <b>כלב השמירה (Cloudflare)</b>\n" +
        problems.map((p) => "• " + p).join("\n") +
        "\n\nהשרשרת בענן לא רצה כצפוי. אפשרויות: להריץ ידנית מלשונית Actions, " +
        "או לבקש מקלוד לטפל.";
      await fetch(
        `https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            chat_id: env.TELEGRAM_CHAT_ID,
            text,
            parse_mode: "HTML",
          }),
        }
      );
    }
  },
};
