@echo off
cd /d %~dp0
rem Read with homography correction (PC only). First run asks for clicks on the reference and live images.
python hom_check.py --xiao http://192.168.11.5 %*
echo EXIT=%ERRORLEVEL%
pause
