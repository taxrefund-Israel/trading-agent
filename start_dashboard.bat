@echo off
cd /d "C:\Users\yaniv\קלוד\trading-agent"
echo Starting trading dashboard...
echo Open browser at http://localhost:8501
echo Do NOT close this window!
echo.
python -m streamlit run signal_agent.py
pause
