@echo off
echo ========================================
echo AI Financial Intelligence Platform
echo ========================================
echo.
echo Starting application...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    py -3.11 -m venv venv
)

REM Install/upgrade dependencies
venv\Scripts\python.exe -m pip install --upgrade pip --quiet
venv\Scripts\python.exe -m pip install -r requirements.txt --quiet

REM Run the application
echo.
echo Launching Streamlit application...
echo Navigate to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

venv\Scripts\python.exe -m streamlit run app.py

pause
