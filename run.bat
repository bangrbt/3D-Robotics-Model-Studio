@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "import mediapipe, cv2, PIL" >nul 2>&1
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" drawing_app.py %*
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Khong chay duoc. Xem thong bao loi phia tren va README.md.
pause
exit /b 1
