@echo off
cd /d "%~dp0"
echo Inchid instanta veche...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0restart.ps1" -Background
exit /b 0
