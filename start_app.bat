@echo off
echo ================================================
echo   Political Bias Detector - Starting...
echo ================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate

REM Start the API server
echo Starting API server on http://localhost:5000...
start cmd /k python src/api.py

REM Wait for server to start
timeout /t 3 /nobreak >nul

REM Open the HTML page in browser
echo Opening web interface...
start test_frontend.html

echo.
echo ================================================
echo   Application is running!
echo   API: http://localhost:5000
echo   Interface opened in your browser
echo ================================================
echo.
echo Press any key to stop the API server...
pause >nul

REM This won't actually stop it, but keeps window open