@echo off
rem Stop the demo service (no prompt besides Python's own).
chcp 65001 >nul
cd /d "%~dp0"
python -X utf8 "scripts\launcher.py" --stop
pause