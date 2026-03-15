@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set PORT=5000
set BUILD_ONLY=0
set BUILD_NATIVE=0

:parse_args
if "%1"=="--build-only" ( set BUILD_ONLY=1 & shift & goto parse_args )
if "%1"=="--port"       ( set PORT=%2      & shift & shift & goto parse_args )

echo.
echo  SIMD Image Processing Engine v2.1
echo  ====================================
echo.

echo [1/3] Installing Python dependencies...
python -m pip install -r requirements.txt -q
if errorlevel 1 (
    echo [ERROR] pip install failed.
    pause
    exit /b 1
)
echo       OK

echo [2/3] Detecting C++ toolchain...

where cmake >nul 2>&1
if errorlevel 1 (
    echo       cmake not found - skipping C++ build
    goto skip_cpp
)

python -c "import pybind11" >nul 2>&1
if errorlevel 1 (
    echo       Installing pybind11...
    python -m pip install pybind11 -q
)

:: Get pybind11 cmake dir
set "TMPD=%TEMP%\pb11dir.txt"
python -c "import pybind11; print(pybind11.get_cmake_dir())" >"%TMPD%" 2>nul
set /p PB11_DIR=<"%TMPD%"
del "%TMPD%" >nul 2>&1
if not defined PB11_DIR ( echo       pybind11 cmake dir not found & goto skip_cpp )
echo       pybind11 dir: !PB11_DIR!

:: Get Python exe
set "TMPP=%TEMP%\pyexe.txt"
python -c "import sys; print(sys.executable)" >"%TMPP%" 2>nul
set /p PYTHON_EXE=<"%TMPP%"
del "%TMPP%" >nul 2>&1
echo       Python exe: !PYTHON_EXE!

:: Get absolute src dir (where the bat file lives)
set SRC_DIR=%~dp0src
echo       Src dir: !SRC_DIR!

:: Detect generator
set GENERATOR=
set "TMPG=%TEMP%\cmgens.txt"
cmake --help >"%TMPG%" 2>&1
findstr /c:"Visual Studio 18 2026" "%TMPG%" >nul 2>&1
if not errorlevel 1 ( set "GENERATOR=Visual Studio 18 2026" & goto gen_found )
findstr /c:"Visual Studio 17 2022" "%TMPG%" >nul 2>&1
if not errorlevel 1 ( set "GENERATOR=Visual Studio 17 2022" & goto gen_found )
findstr /c:"Visual Studio 16 2019" "%TMPG%" >nul 2>&1
if not errorlevel 1 ( set "GENERATOR=Visual Studio 16 2019" & goto gen_found )
where ninja >nul 2>&1
if not errorlevel 1 ( set "GENERATOR=Ninja" & goto gen_found )
where nmake >nul 2>&1
if not errorlevel 1 ( set "GENERATOR=NMake Makefiles" & goto gen_found )
echo       No generator found.
del "%TMPG%" >nul 2>&1
goto skip_cpp

:gen_found
del "%TMPG%" >nul 2>&1
echo       Generator: !GENERATOR!

:: Always delete old build dir to force fresh configure
echo       Cleaning old build dir...
if exist build rmdir /s /q build
mkdir build

if "!GENERATOR!"=="Ninja" (
    cmake -B build -G "Ninja" ^
        -DCMAKE_BUILD_TYPE=Release ^
        -Dpybind11_DIR="!PB11_DIR!" ^
        -DPython3_EXECUTABLE="!PYTHON_EXE!" ^
        -DCMAKE_INCLUDE_PATH="!SRC_DIR!"
    if errorlevel 1 ( echo       Configure failed & goto skip_cpp )
    cmake --build build
    goto copy_pyd
)
if "!GENERATOR!"=="NMake Makefiles" (
    cmake -B build -G "NMake Makefiles" ^
        -DCMAKE_BUILD_TYPE=Release ^
        -Dpybind11_DIR="!PB11_DIR!" ^
        -DPython3_EXECUTABLE="!PYTHON_EXE!" ^
        -DCMAKE_INCLUDE_PATH="!SRC_DIR!"
    if errorlevel 1 ( echo       Configure failed & goto skip_cpp )
    cmake --build build
    goto copy_pyd
)

cmake -B build -G "!GENERATOR!" -A x64 ^
    -Dpybind11_DIR="!PB11_DIR!" ^
    -DPython3_EXECUTABLE="!PYTHON_EXE!" ^
    -DCMAKE_INCLUDE_PATH="!SRC_DIR!"
if errorlevel 1 ( echo       Configure failed & goto skip_cpp )
cmake --build build --config Release

:copy_pyd
:: Copy .pyd from build subdirs to project root so Python can import it
for /r build %%f in (simd_engine*.pyd) do (
    echo       Copying %%f
    copy /y "%%f" "." >nul 2>&1
)

:check_pyd
set BUILD_NATIVE=0
for %%f in (simd_engine*.pyd) do (
    echo       Built: %%f
    set BUILD_NATIVE=1
)
if "%BUILD_NATIVE%"=="0" echo       No .pyd found - Python reference mode
goto after_cpp

:skip_cpp
echo       Running in Python reference mode

:after_cpp
echo.
if "%BUILD_ONLY%"=="1" ( echo Build done. & exit /b 0 )

echo [3/3] Starting server...
echo.
echo  Native C++  : %BUILD_NATIVE%
echo  URL         : http://localhost:%PORT%
echo  Stop        : Ctrl+C
echo.

set PORT=%PORT%
python server.py
if errorlevel 1 (
    echo.
    echo [ERROR] server.py exited with an error - see above for details.
    pause
)
