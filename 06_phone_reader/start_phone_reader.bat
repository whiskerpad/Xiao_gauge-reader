@echo off
cd /d %~dp0
rem iPhone gauge reader: open the QR shown in the PC browser with the iPhone camera.
rem Uses the certificate of G:\iphone_capture (already trusted on the iPhone). Ports: https 8444, PC page 8766.
python -c "import torch, cv2, PIL" 2>NUL || python -m pip install torch opencv-python pillow numpy
python phone_gauge_server.py %*
pause
