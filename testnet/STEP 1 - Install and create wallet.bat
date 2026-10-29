@echo off
cd /d "%~dp0"
set "PY=%USERPROFILE%\anaconda3\python.exe"
if not exist "%PY%" set "PY=python"
echo Installing the required Python packages (one time only)...
"%PY%" -m pip install -r requirements.txt
echo.
"%PY%" 00_create_wallet.py
echo.
echo Copy the address above (it starts with 0x) and get free Sepolia ETH from a faucet - see README.md.
pause
