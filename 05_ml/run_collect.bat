@echo off
cd /d %~dp0
rem Collect 2000 positions, front camera, ceiling light (about 1.5-2 h)
python collect.py --port COM8 --xiao http://192.168.11.5 --count 2000 --tag front_ceiling
echo EXIT=%ERRORLEVEL%
pause
