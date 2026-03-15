@echo off
cd /d "%~dp0"

echo Patching server.py ...

:: Backup original
copy /y server.py server.py.bak >nul

:: Replace static_folder="frontend" with static_folder="."
powershell -NoProfile -Command ^
  "(Get-Content server.py -Raw) -replace 'static_folder=""frontend""', 'static_folder="".""' | Set-Content server.py -Encoding UTF8"

if errorlevel 1 (
    echo [FAIL] PowerShell patch failed.
    pause
    exit /b 1
)

:: Verify
powershell -NoProfile -Command ^
  "if ((Get-Content server.py -Raw) -match 'static_folder=""\.""") { Write-Host 'OK - patch applied' } else { Write-Host 'FAIL - patch not found'; exit 1 }"

echo.
echo Starting server...
echo.
python server.py
if errorlevel 1 (
    echo.
    echo [ERROR] server.py crashed - see above.
    pause
)
