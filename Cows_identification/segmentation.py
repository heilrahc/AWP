import sys
from collections import defaultdict
import numpy as np
import shutil
import cv2
import os
import torch

sys.path.append("..")
TRAIN_PATH = '/home/mine01/Desktop/code/AWP/Cows_identification/data/cows_datasets/train'
TEST_PATH = '/home/mine01/Desktop/code/AWP/Cows_identification/data/cows_datasets/test'
video_dict = defaultdict(str)

def find_index_of_class(cls, target=19.):
    equals_target = torch.eq(cls, target)
    if torch.any(equals_target):
        return torch.nonzero(equals_target)[0].item()
    else:
        return None

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


def process_image(image_path, train_path, test_path, seg_model, test_ratio=0.0):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    seg_result = seg_model(image)[0]
    indx = find_index_of_class(seg_result.boxes.cls)

    if indx is not None:
        seg_mask = seg_result.masks.data[indx].cpu().numpy()

        # Resize the image to match the mask dimensions
        resized_mask = cv2.resize(seg_mask, (image.shape[1], image.shape[0]))

        # Expand the dimensions of the mask to match the number of channels in the resized image
        seg_mask_expanded = np.expand_dims(resized_mask, axis=2)
        seg_mask_expanded = np.tile(seg_mask_expanded, (1, 1, image.shape[2]))

        # Perform the multiplication
        seg1 = seg_mask_expanded * image

        # convert seg1 back to RGB color scheme
        masked_image = cv2.cvtColor(seg1.astype(np.uint8), cv2.COLOR_RGB2BGR)
        masked_image = auto_crop(masked_image)

        cow_id, video_id, _, _ = os.path.basename(image_path).split('_')
        if not video_dict[f"{cow_id}_{video_id}"]:  # Check if this video has been assigned before
            video_dict[f"{cow_id}_{video_id}"] = test_path if np.random.rand() < test_ratio else train_path
        new_dir = os.path.join(video_dict[f"{cow_id}_{video_id}"], cow_id)
        os.makedirs(new_dir, exist_ok=True)
        cv2.imwrite(os.path.join(new_dir, f'mask_{os.path.basename(image_path)}'), masked_image)


def segment_images(frame_dir, seg_model):
    create_directories(TRAIN_PATH)
    create_directories(TEST_PATH)
    for image_file in os.listdir(frame_dir):
        image_path = os.path.join(frame_dir, image_file)
        print(image_path)
        if os.path.isfile(image_path):
            process_image(image_path, TRAIN_PATH, TEST_PATH, seg_model)
