@echo off
cd /d %~dp0
rem iPhone + XIAO, 5 poses x 200 positions. Move iPhone and press Enter at each pose.
python collect_iphone.py --port COM8 --xiao http://192.168.11.5 --poses 5 --count 200 --tag iphone_ceiling
echo EXIT=%ERRORLEVEL%
pause
