from ultralytics import YOLO
import cv2
import os

data_path = "/home/mine01/Desktop/code/AWP/Cows_identification/cows_detect_data.yaml"
model = YOLO('yolov8m.pt')

# model.train(data=data_path, epochs=50, imgsz=640)

results = model('/home/mine01/Desktop/code/AWP/Cows_identification/videos/4008/010421_MVI_0114_short_22.MP4', save=True)