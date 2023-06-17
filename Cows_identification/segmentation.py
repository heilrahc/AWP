import sys
from segment_anything import sam_model_registry, SamPredictor
from collections import defaultdict
import numpy as np
import shutil
import cv2
import os

sys.path.append("..")
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda"
TRAIN_PATH = '/home/mine01/Desktop/code/AWP/Cows_identification/cows_datasets/train'
TEST_PATH = '/home/mine01/Desktop/code/AWP/Cows_identification/cows_datasets/test'
video_dict = defaultdict(str)


def init_sam_model(model_type, device, checkpoint):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


def create_directories(path):
    os.makedirs(path, exist_ok=True)

    # Delete all existing files in the frame directory
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):  # If there are subdirectories, and you want to remove them as well
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def auto_crop(image):
    # Load image
    img = image

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding the image
    _, thresh_img = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the threshold image
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small irrelevant contours based on area
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    # If there are no contours, return original image
    if not contours:
        print("No contours found")
        return img

    # Find bounding box coordinates
    x, y, w, h = cv2.boundingRect(max(contours, key = cv2.contourArea))

    # Crop the original image to the found coordinates
    crop_img = img[y:y+h, x:x+w]

    return crop_img


def process_image(image_path, train_path, test_path, predictor, test_ratio=0.1):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)
    input_point = np.array([[1000, 630], [950, 570], [1060, 600]])
    # input_point = np.array([[600, 500], [630, 530], [570, 470]])
    input_label = np.array([1, 1, 1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    mask_input = logits[np.argmax(scores), :, :]
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )
    masked_image = image * masks[0][:, :, None]  # If masks has more than 1 dimension, select the relevant one
    masked_image = cv2.cvtColor(masked_image.astype(np.uint8), cv2.COLOR_RGB2BGR)  # Convert the masked image back to BGR color scheme for saving
    masked_image = auto_crop(masked_image)

    cow_id, video_id, _, _ = os.path.basename(image_path).split('_')
    if not video_dict[f"{cow_id}_{video_id}"]:  # Check if this video has been assigned before
        video_dict[f"{cow_id}_{video_id}"] = test_path if np.random.rand() < test_ratio else train_path
    new_dir = os.path.join(video_dict[f"{cow_id}_{video_id}"], cow_id)
    os.makedirs(new_dir, exist_ok=True)
    cv2.imwrite(os.path.join(new_dir, f'mask_{os.path.basename(image_path)}'), masked_image)


def segment_images(frame_dir):
    predictor = init_sam_model(MODEL_TYPE, DEVICE, SAM_CHECKPOINT)
    create_directories(TRAIN_PATH)
    create_directories(TEST_PATH)
    for image_file in os.listdir(frame_dir):
        image_path = os.path.join(frame_dir, image_file)
        print(image_path)
        if os.path.isfile(image_path):
            process_image(image_path, TRAIN_PATH, TEST_PATH, predictor)
