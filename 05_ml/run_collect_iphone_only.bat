@echo off
cd /d %~dp0
rem iPhone only (no XIAO), 5 poses x 200 positions. Move iPhone and press Enter at each pose.
python collect_iphone.py --port COM8 --poses 5 --count 200 --tag iphone_only
echo EXIT=%ERRORLEVEL%
pause
