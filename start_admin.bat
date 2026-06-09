@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process py -ArgumentList '-3','main.py' -WorkingDirectory '%~dp0' -Verb RunAs"
