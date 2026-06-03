"""
מפעיל את signal_agent.py עם localhost.run tunnel ציבורי.
ללא צורך בהרשמה — עובד דרך SSH מובנה.
הרצה: python launch.py
"""
import subprocess
import sys
import time
import re
import threading

STREAMLIT_PORT = 8501

def stream_output(proc):
    for line in proc.stdout:
        print(line, end="")

# ── הפעל Streamlit ────────────────────────────────────────────
print("מפעיל Streamlit...")
streamlit_proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "signal_agent.py",
     "--server.port", str(STREAMLIT_PORT),
     "--server.headless", "true"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
)
threading.Thread(target=stream_output, args=(streamlit_proc,), daemon=True).start()
time.sleep(4)

# ── פתח tunnel דרך localhost.run ──────────────────────────────
print("פותח tunnel ציבורי...")
tunnel_proc = subprocess.Popen(
    ["ssh", "-o", "StrictHostKeyChecking=no",
     "-R", f"80:localhost:{STREAMLIT_PORT}",
     "nokey@localhost.run"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, encoding="utf-8", errors="replace"
)

url_found = False
try:
    for line in tunnel_proc.stdout:
        line = line.strip()
        if line:
            print(line)
        match = re.search(r"https://[a-zA-Z0-9\-]+\.lhr\.life", line)
        if match and not url_found:
            url_found = True
            public_url = match.group(0)
            print()
            print("=" * 55)
            print(f"  >> הקישור הציבורי שלך:")
            print(f"  {public_url}")
            print("=" * 55)
            print("  שלח קישור זה לכל מחשב בעולם.")
            print("  הקישור פעיל כל עוד חלון זה פתוח.")
            print("  לסגירה: Ctrl+C")
            print()
except KeyboardInterrupt:
    pass
finally:
    print("\nסוגר...")
    tunnel_proc.terminate()
    streamlit_proc.terminate()
