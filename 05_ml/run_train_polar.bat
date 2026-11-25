@echo off
cd /d %~dp0
rem Build training data and train the polar CNN (for XIAO), then write the C header.
rem usage: run_train_polar.bat data\20261119_074420_front_ceiling
rem NOTE: overwrites ..\02_xiao_reader\include\nn_polar_data.h (the old one is kept as nn_polar_data.h.bak)
set SRC=%1
if "%SRC%"=="" set SRC=data\20261119_074420_front_ceiling
python -c "import torch" 2>NUL || python -m pip install torch numpy opencv-python pillow
if not exist models\reg_ref.npy python make_npz.py ref %SRC%
python make_npz.py polar %SRC% --name front
python train_polar.py --train front --epochs 60 --out models\my_polar.pt
copy /y ..\02_xiao_reader\include\nn_polar_data.h ..\02_xiao_reader\include\nn_polar_data.h.bak >NUL
python export_polar.py models\my_polar.pt models\reg_ref.npy ..\02_xiao_reader\include\nn_polar_data.h
echo EXIT=%ERRORLEVEL%
pause
