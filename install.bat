@echo off
echo Instalare Key Auto...
cd /d "%~dp0"
py -3 -m pip install -r requirements.txt || python -m pip install -r requirements.txt
echo.
echo Gata. Ruleaza start.bat pentru a porni aplicatia.
pause
