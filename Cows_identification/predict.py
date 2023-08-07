from ultralytics import YOLO


def init_yolo(yolo_cls, train_dataset, epochs, model_path, re_train):
    if model_path is None:
        # Load a pretrained YOLO model (recommended for training)
        yolo = yolo_cls

        # Train the model
        yolo.train(data=train_dataset, epochs=epochs, imgsz=640)
    else:
        # Load a trained YOLO model
        yolo = YOLO(model_path)

        if re_train:
            # Re-Train the model
            yolo.train(data=train_dataset, epochs=epochs, imgsz=640)



    # Validate the model
    # metrics = yolo.val()
    # print(metrics.top1)   # top1 accuracy
    # print(metrics.top5)   # top5 accuracy

    return yolo


def yolo_train(yolo_cls, train_dataset, epochs, yolo_cls_trained_path=None, train=False):
    yolo = init_yolo(yolo_cls, train_dataset, epochs, yolo_cls_trained_path, re_train=train)
    return yolo


#
# def recreate_directory_structure(input_dir, output_dir):
#     # Recreate the same directory structure in output directory
#     for root, dirs, files in os.walk(input_dir):
#         for dir_ in dirs:
#             os.makedirs(os.path.join(output_dir, os.path.relpath(os.path.join(root, dir_), input_dir)), exist_ok=True)
#
#
# def classify_and_save_results(yolo, input_dir, output_dir):
#     # Classify each image and save results
#     for root, dirs, files in os.walk(input_dir):
#         for file in files:
#             if file.endswith(('.jpg', '.png', '.jpeg')):  # add or remove file extensions as required
#                 image_path = os.path.join(root, file)
#                 result = yolo(image_path)
#                 res_plotted = result[0].plot()
#
#                 # Generate a save path with a new name
#                 output_path = os.path.join(output_dir, os.path.relpath(root, input_dir), f'{file}')
#                 cv2.imwrite(output_path, res_plotted)

