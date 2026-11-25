@echo off
cd /d %~dp0
rem Check framing: save one image to data\check.jpg
python collect.py --port COM8 --xiao http://192.168.11.5 --check
echo EXIT=%ERRORLEVEL%
pause
