@echo off
cd /d %~dp0
rem XIAO live image + training reference overlay (for re-positioning the camera)
python align_view.py --xiao http://192.168.11.5
echo EXIT=%ERRORLEVEL%
pause
