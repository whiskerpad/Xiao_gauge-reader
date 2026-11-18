@echo off
cd /d %~dp0
python -c "import flask" 2>NUL || python -m pip install flask requests
python app.py --xiao http://192.168.11.5 --interval 5 --n 3
pause
