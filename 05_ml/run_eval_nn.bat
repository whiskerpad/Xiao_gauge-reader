echo off
cd /d %~dp0
rem Evaluate CNN reading (/nn) at 50 new random positions.
rem The summary also shows results for several ensemble thresholds (computed from the same images).
set /p L=Lighting memo (letters only, e.g. sun, ceiling, dark, desk): 
python eval_nn.py --port COM8 --xiao http://192.168.11.5 --count 50 --tag nn_pol_%L%
echo EXIT=%ERRORLEVEL%
pause
