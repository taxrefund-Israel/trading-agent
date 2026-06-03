"""
מפעיל Streamlit כ-process מנותק לחלוטין מהחלון הנוכחי.
"""
import subprocess, sys, os, time, socket

PYTHON = sys.executable
APP    = os.path.join(os.path.dirname(__file__), "signal_agent.py")
PORT   = 8501

DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000

def port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

if port_open(PORT):
    print(f"Streamlit כבר רץ על http://localhost:{PORT}")
    sys.exit(0)

proc = subprocess.Popen(
    [PYTHON, "-m", "streamlit", "run", APP,
     "--server.port", str(PORT),
     "--server.headless", "true"],
    creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
    close_fds=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL,
)

print(f"מפעיל... (PID {proc.pid})")
for i in range(15):
    time.sleep(1)
    if port_open(PORT):
        print(f"מוכן! http://localhost:{PORT}")
        sys.exit(0)

print("לא הצליח להפעיל, בדוק שגיאות.")
sys.exit(1)
