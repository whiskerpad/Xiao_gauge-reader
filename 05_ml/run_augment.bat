@echo off
rem Augment training images by artificial tilt / rotation / lighting
rem usage: run_augment.bat data\20261119_074420_front_ceiling 5
cd /d %~dp0
set SRC=%1
set CNT=%2
if "%SRC%"=="" set SRC=data\20261119_074420_front_ceiling
if "%CNT%"=="" set CNT=5
python augment_dataset.py %SRC% %SRC%_aug --count %CNT%
echo EXIT=%ERRORLEVEL%
pause
