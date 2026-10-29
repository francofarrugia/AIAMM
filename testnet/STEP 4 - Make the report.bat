@echo off
cd /d "%~dp0"
set "PY=%USERPROFILE%\anaconda3\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" 03_report.py
echo.
echo Done. Tell Claude that step 4 has finished.
pause
