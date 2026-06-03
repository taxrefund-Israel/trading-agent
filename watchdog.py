"""
watchdog.py — מוודא שה-Streamlit תמיד רץ.
מופעל בעת כניסה ל-Windows (דרך תיקיית Startup).
"""
import subprocess, sys, os, time, socket

PYTHON = sys.executable
APP    = os.path.join(os.path.dirname(__file__), "signal_agent.py")
PORT   = 8501
CHECK_INTERVAL = 60  # בדוק כל דקה

DETACHED = 0x00000008
NO_WIN   = 0x08000000

def port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def start_streamlit():
    subprocess.Popen(
        [PYTHON, "-m", "streamlit", "run", APP,
         "--server.port", str(PORT),
         "--server.headless", "true"],
        creationflags=DETACHED | NO_WIN,
        close_fds=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )

# הפעלה ראשונה
if not port_open(PORT):
    start_streamlit()
    time.sleep(12)

# לולאת watchdog — רץ ברקע ומוודא שהאפליקציה חיה
while True:
    time.sleep(CHECK_INTERVAL)
    if not port_open(PORT):
        start_streamlit()
        time.sleep(12)
