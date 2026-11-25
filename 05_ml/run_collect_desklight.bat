@echo off
cd /d %~dp0
rem Lighting set: Desk light from an oblique angle (glare on the cover glass)
rem 500 positions (20% near 0 or 1.0), about 40 min
python collect.py --port COM8 --xiao http://192.168.11.5 --count 500 --edge 0.2 --tag light_desklight
echo EXIT=%ERRORLEVEL%
pause
