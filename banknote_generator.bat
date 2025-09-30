@echo off
chcp 65001

REM ==============================================
REM Simple launcher for main.py
REM ==============================================

REM Call main.py with all arguments passed to this batch file
python main.py %*

pause
exit /b
