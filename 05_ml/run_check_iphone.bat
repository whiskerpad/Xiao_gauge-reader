@echo off
cd /d %~dp0
rem Check iPhone: take one shot to data\check_iphone.jpg (start iphone_capture first)
python collect_iphone.py --port COM8 --check
echo EXIT=%ERRORLEVEL%
pause
