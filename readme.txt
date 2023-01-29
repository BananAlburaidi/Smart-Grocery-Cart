To train a custom model

https://www.youtube.com/watch?v=M0EA4u1x4fs&t=263s

1- Using cmd cd open the directory of the smart shopping cart
2- pip install -r requirements.txt

to run the smart shopping cart object detection:
For WEBCAM:
python detect.py --weights best.pt --source 0

For ESP32-CAM

python detect.py --weights best.pt --source "192.168.X.X/capture"



