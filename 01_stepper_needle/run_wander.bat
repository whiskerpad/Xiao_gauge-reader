@echo off
cd /d %~dp0
python needle_wander.py --port COM8 --server http://127.0.0.1:5000
pause
