@echo off
rem Run the livestacker from its .venv. Pass through any arguments, e.g.
rem   livestack.bat --watch "C:\path\to\folder" -v
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo error: run install.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m livestacker %*
