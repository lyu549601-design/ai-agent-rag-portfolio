@echo off
rem ============================================================
rem  Trusted Procurement Agent - one-click launcher
rem  Double-click this file:
rem    - not running      -> start the web demo and open the browser
rem    - already running  -> ask whether to shut it down
rem  User-facing messages are printed by Python (UTF-8), so this
rem  file stays ASCII-only to avoid codepage issues.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo   [ERROR] Python not found in PATH.
  echo   Install Python 3.11+ from https://www.python.org/downloads/
  echo   and make sure "Add python.exe to PATH" is checked.
  echo.
  pause
  exit /b 1
)

python -X utf8 "scripts\launcher.py" %*
if errorlevel 1 pause