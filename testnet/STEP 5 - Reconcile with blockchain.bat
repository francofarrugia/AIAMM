@echo off
cd /d "%~dp0"
where py >nul 2>nul && (py 04_reconcile.py) || (python 04_reconcile.py)
echo.
echo Done. Tell Claude "done".
pause
