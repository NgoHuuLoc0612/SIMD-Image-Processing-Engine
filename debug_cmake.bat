@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo === DEBUG: cmake detection ===

where cmake
if errorlevel 1 (
    echo cmake NOT found in PATH
    pause
    exit /b 1
)

echo.
echo === cmake version ===
cmake --version

echo.
echo === cmake available generators (look for Visual Studio lines) ===
cmake --help 2>&1 | findstr /i /c:"visual studio" /c:"ninja" /c:"nmake" /c:"Generators"

echo.
echo === Testing VS 18 2026 ===
cmake -E echo test >nul
cmake -B _test18 -G "Visual Studio 18 2026" -A x64 >_test18.log 2>&1
echo errorlevel: %errorlevel%
type _test18.log

echo.
echo === Testing VS 17 2022 ===
cmake -B _test17 -G "Visual Studio 17 2022" -A x64 >_test17.log 2>&1
echo errorlevel: %errorlevel%
type _test17.log

echo.
echo === Cleaning up ===
if exist _test18 rmdir /s /q _test18
if exist _test17 rmdir /s /q _test17
if exist _test18.log del _test18.log
if exist _test17.log del _test17.log

pause
