@echo off
cd /d %~dp0
python run_eval.py --port COM8 --xiao http://192.168.11.5 --mode random --count 50 --tag mk_ring2 --save-all
echo EXIT=%ERRORLEVEL%
pause
