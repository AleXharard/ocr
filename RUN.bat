@echo off
setlocal EnableExtensions
title Key Auto

cd /d "%~dp0"

:: ── 1. Self-elevate (tastele in joc necesita admin) ───────────────────────
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Cer drepturi de administrator...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

cd /d "%~dp0"

:: ── 2. EXE unic (fara consola) ────────────────────────────────────────────
if exist "dist\KeyAuto.exe" (
    start "" "%~dp0dist\KeyAuto.exe"
    exit /b 0
)

:: compat: build vechi cu folder
if exist "dist\KeyAuto\KeyAuto.exe" (
    start "" /D "%~dp0dist\KeyAuto" "%~dp0dist\KeyAuto\KeyAuto.exe"
    exit /b 0
)

:: ── 3. Fallback: sursa Python ─────────────────────────────────────────────
where py >nul 2>&1 && (set "PY=py -3") || (set "PY=python")
%PY% --version >nul 2>&1
if errorlevel 1 (
    echo Python negasit. Ruleaza BUILD.bat pentru dist\KeyAuto.exe
    pause
    exit /b 1
)

%PY% -c "import customtkinter, cv2, mss, numpy, dxcam, rapidocr_onnxruntime, pydirectinput, keyboard, PIL" >nul 2>&1
if errorlevel 1 (
    echo Instalez dependintele...
    %PY% -m pip install -r requirements.txt
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0restart.ps1" -Background
exit /b 0
