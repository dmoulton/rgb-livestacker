@echo off
rem One-time setup for the livestacker on Windows.
rem Double-click this file, or run it from a command prompt.
setlocal
cd /d "%~dp0"

set "PY="
for %%c in ("py -3.13" "py -3.12" "py -3.11" "py -3" "python") do (
    %%~c -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY=%%~c"
        goto :found
    )
)
echo error: need Python 3.11 or newer.
echo Download it from https://www.python.org/downloads/windows/
echo and tick "Add python.exe to PATH" in the installer, then run this again.
pause
exit /b 1

:found
echo Using %PY%

if not exist ".venv\" (
    %PY% -m venv .venv
    echo Created .venv
)

set PIP_USER=0
".venv\Scripts\python.exe" -m pip install --upgrade --quiet pip
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo.
    echo Install failed - see the messages above.
    pause
    exit /b 1
)

echo.
echo Done. Double-click livestack.bat to open the launcher window,
echo or run it with options:
echo   livestack.bat --watch  "C:\path\to\NINA\Images"      ^(headless, live^)
echo   livestack.bat --replay "C:\path\to\old\session"      ^(headless, test^)
pause
