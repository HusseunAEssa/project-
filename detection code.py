# Real-Time (AI) for Unmanned aerial vehicles (UAVs) Object Detection.
from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(1)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)
#cap = cv2.VideoCapture("ppe-2.mp4")  # For Video

model = YOLO("../Yolo-Weights/kaggle_n++.pt")
#cap = cv2.VideoCapture("images/11.mp4")

classNames = ['Drone', 'drone', 'flock of birds']

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True,conf=0.64)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # هذا على النربع
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            label = f'{classNames[cls]} {conf}%'
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255,255,0), 1)

            # طباعة نسبة الثقة في نافذة الكوماند
            print(f"{classNames[cls]}: {conf}%")
       # next_detection_time = new_frame_time + detection_interval
        #fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
       # print(f"FPS: {fps}")
        # Display FPS and label on the image
        if len(boxes) > 0:  # Make sure there's at least one box detected
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
        # مال الطباعة الي تطلع ع الشاشه

    cv2.imshow("Image", img)
    cv2.waitKey(1)
