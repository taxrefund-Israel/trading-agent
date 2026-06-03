@echo off
cd /d "C:\Users\yaniv\קלוד\trading-agent"
"C:\Users\yaniv\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m streamlit run signal_agent.py --server.port 8501 --server.headless true < nul