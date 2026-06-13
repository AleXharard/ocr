@echo off
setlocal EnableExtensions
title Key Auto OCR - Launcher

cd /d "%~dp0"

:: ── 1. Self-elevate to administrator ───────────────────────────────────────
:: Sending keys into a game (pydirectinput / keyboard) needs admin rights.
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Cer drepturi de administrator...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

cd /d "%~dp0"

:: ── 2. Pick the available Python command ───────────────────────────────────
where py >nul 2>&1 && (set "PY=py -3") || (set "PY=python")

%PY% --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [EROARE] Python nu a fost gasit. Instaleaza Python 3 si reincearca.
    echo.
    pause
    exit /b 1
)

:: ── 3. Ensure dependencies are installed (fast import check) ────────────────
%PY% -c "import customtkinter, cv2, mss, numpy, dxcam, rapidocr_onnxruntime, pydirectinput, keyboard, PIL" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Instalez dependintele... ^(o singura data^)
    echo.
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [EROARE] Instalarea dependintelor a esuat. Vezi mesajele de mai sus.
        echo.
        pause
        exit /b 1
    )
)

:: ── 4. Launch the app ──────────────────────────────────────────────────────
echo.
echo Pornesc Key Auto OCR...
echo.
%PY% main.py

echo.
echo Aplicatia s-a inchis.
pause
