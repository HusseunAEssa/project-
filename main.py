import cv2
from ultralytics import YOLO


model = YOLO("../Yolo-Weights/auto.pt")
results = model("C:/Users/HAZ/Desktop/New folder/5146_jpg.rf.a961d1a4585e3f56bc5c7245ebbebf14.jpg", show=True)

cv2.waitKey(0)
