@echo off
cd /d "C:\Users\yaniv\קלוד\trading-agent"
"C:\Users\yaniv\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m streamlit run us_dashboard.py --server.port 8502 --server.headless true
