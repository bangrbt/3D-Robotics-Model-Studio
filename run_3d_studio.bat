@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo   KHOI DONG 3D ROBOTICS MODEL STUDIO
echo ========================================================
if not exist ".venv\Scripts\python.exe" (
    echo Dang tao moi truong ao .venv...
    python -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "import trimesh, numpy" >nul 2>&1
if errorlevel 1 (
    echo Dang cai dat thu vien 3D (trimesh, numpy)...
    ".venv\Scripts\python.exe" -m pip install trimesh numpy scipy
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" run_3d_pipeline.py %*
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Khong chay duoc 3D Studio. Vui long kiem tra Python va loi phia tren.
pause
exit /b 1
