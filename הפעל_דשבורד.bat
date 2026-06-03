@echo off
cd /d "C:\Users\yaniv\קלוד\trading-agent"
echo מפעיל דשבורד איתותי מסחר...
echo פתח את הדפדפן על http://localhost:8501
echo אל תסגור חלון זה!
echo.
python -m streamlit run signal_agent.py
pause
