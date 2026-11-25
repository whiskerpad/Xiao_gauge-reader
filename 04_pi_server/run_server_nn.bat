@echo off
cd /d %~dp0
rem Trend display with the machine-learning reader (polar CNN). Data goes to data\gauge_nn.db
python -c "import flask" 2>NUL || python -m pip install flask requests
python app.py --xiao http://192.168.11.5 --interval 10 --n 3 --nn 1 --db data\gauge_nn.db
pause
