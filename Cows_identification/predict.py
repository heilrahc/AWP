from ultralytics import YOLO
import cv2
import os


yolo_data = "/home/mine01/Desktop/code/AWP/Cows_identification/cows_datasets_2"
# Load a pretrained YOLO model (recommended for training)
# yolo = YOLO('yolov8m-cls.pt')

# this model has a validation accuracy of
# nano: train; xlarge: train2; medium: train5
yolo = YOLO('runs/classify/train5/weights/best.pt')


# # Train the model
# yolo.train(data=yolo_data, epochs=50, imgsz=640)

# Validate the model
metrics = yolo.val()  # no arguments needed, dataset and settings remembered
print(metrics.top1)   # top1 accuracy
print(metrics.top5)   # top5 accuracy

#test_path = "/home/mine01/Desktop/code/AWP/Cows_identification/cows_datasets_2/test/5064"
input_dir = "/home/mine01/Desktop/code/AWP/Cows_identification/cows_datasets_2/test"
output_dir = "/home/mine01/Desktop/code/AWP/Cows_identification/results_m"


# Recreate the same directory structure in output directory
for root, dirs, files in os.walk(input_dir):
    for dir_ in dirs:
        os.makedirs(os.path.join(output_dir, os.path.relpath(os.path.join(root, dir_), input_dir)), exist_ok=True)


# Classify each image and save results
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):  # add or remove file extensions as required
            image_path = os.path.join(root, file)
            result = yolo(image_path)
            res_plotted = result[0].plot()

            # Generate a save path with a new name
            output_path = os.path.join(output_dir, os.path.relpath(root, input_dir), f'{file}')
            # output_path = os.path.join(output_dir, f'{file}')
            cv2.imwrite(output_path, res_plotted)

