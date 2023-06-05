from ultralytics import YOLO
import cv2
import os


yolo_data = "/home/mine01/Desktop/code/AWP/SAM_Clustering/cows_datasets"
# Load a pretrained YOLO model (recommended for training)
yolo = YOLO('runs/classify/train7/weights/best.pt')

print(yolo.summary())

# # Train the model
# yolo.train(data=yolo_data, epochs=50, imgsz=640)

# Validate the model
metrics = yolo.val()  # no arguments needed, dataset and settings remembered
print(metrics.top1)   # top1 accuracy
print(metrics.top5)   # top5 accuracy

test_path_depth = "/home/mine01/Desktop/code/AWP/SAM_Clustering/cows_datasets/test/6062"
test_path = "/home/mine01/Desktop/code/AWP/SAM_Clustering/cows_datasets/test"
result_path = "/home/mine01/Desktop/code/AWP/SAM_Clustering/results"

# Loop through the images in the folder
for image_file in os.listdir(test_path_depth):
    if image_file.endswith('.jpg') or image_file.endswith('.jpeg') or image_file.endswith('.png'):
        image_path = os.path.join(test_path_depth, image_file)
        result = yolo(image_path)
        res_plotted = result[0].plot()

        # Generate a save path with a new name
        save_path = os.path.join(result_path, "result_" + image_file)
        cv2.imwrite(save_path, res_plotted)

