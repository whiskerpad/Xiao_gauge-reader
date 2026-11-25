@echo off
cd /d %~dp0
rem Error measurement with the stepper gauge (UNO on COM8). Close run_wander.bat / other users of COM8 first.
rem The needle stops at each target value; after 10 reads it moves to the next. Summary: python summarize_phone.py
python -c "import torch, cv2, PIL, serial" 2>NUL || python -m pip install torch opencv-python pillow numpy pyserial
python phone_gauge_server.py --needle COM8 --targets 0.2,0.5,0.8 --per 10 %*
pause
