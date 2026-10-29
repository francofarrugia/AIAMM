@echo off
cd /d "%~dp0"
set "PY=%USERPROFILE%\anaconda3\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" 02_run_agent.py --steps 48
echo.
echo If it stopped early, double-click this file again: it resumes where it stopped.
pause
