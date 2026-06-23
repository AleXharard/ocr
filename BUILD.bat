@echo off
setlocal EnableExtensions
title Key Auto — Build EXE

cd /d "%~dp0"

where py >nul 2>&1 && (set "PY=py -3") || (set "PY=python")
%PY% --version >nul 2>&1
if errorlevel 1 (
    echo [EROARE] Python nu a fost gasit.
    pause
    exit /b 1
)

if not exist "glyph_bank.npz" (
    echo [EROARE] Lipseste glyph_bank.npz — ruleaza build_bank.py mai intai.
    pause
    exit /b 1
)

if not exist "poze mina\piatra.png" (
    echo [EROARE] Lipseste poze mina\piatra.png
    pause
    exit /b 1
)

echo Instalez PyInstaller daca e nevoie...
%PY% -m pip install pyinstaller>=6.0 -q

echo.
echo Construiesc UN SINGUR KeyAuto.exe ^(fara consola, cu admin^)...
echo Poate dura 1-2 minute...
echo.
%PY% -m PyInstaller KeyAuto.spec --noconfirm --clean
if errorlevel 1 (
    echo.
    echo [EROARE] Build esuat.
    pause
    exit /b 1
)

echo.
echo Gata: dist\KeyAuto.exe
echo Trimite DOAR acest fisier. Config ^(regions.json^) si loguri apar langa exe la prima rulare.
echo.
pause
