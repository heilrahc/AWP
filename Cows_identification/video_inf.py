import os
import cv2
from segment_anything import sam_model_registry, SamPredictor
from collections import defaultdict
import numpy as np
import json
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil

SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda"
TEST_VIDEOS_PATH = '/home/mine01/Desktop/code/AWP/Cows_identification/test_videos'


def init_sam_model(model_type, device, checkpoint):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


def extract_frames(video_path, num_frames, time_interval):
    # frames
    frames = []

    # Load the video using OpenCV
    vidcap = cv2.VideoCapture(video_path)

    # Get video frames per second (fps)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    # Calculate the frame skip based on the desired time interval
    frame_skip = int(fps * time_interval)

    # Get total frames
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    # Calculate the start and end frame for the middle few 5%
    start_frame = int(total_frames * 0.45)
    end_frame = int(total_frames * 0.55)

    success, image = vidcap.read()
    frame_count = 0
    extracted_frames = 0
    while success:
        # Check if this frame is one of the frames we want to extract
        if frame_count % frame_skip == 0 and extracted_frames < num_frames and start_frame <= frame_count <= end_frame:
            frames.append(image)

            extracted_frames += 1
        success, image = vidcap.read()
        frame_count += 1

    return frames


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


def process_image(frames, predictor):
    masked_images = []
    for frame in frames:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

        masked_images.append(masked_image)

    return masked_images


def yolo_predict(video_path, masked_images, yolo):
    dict = {}
    for image in masked_images:
        result = yolo(image)
        probs = result[0].probs
        top5_indx = probs.top5
        top5_class = [result[0].names[i] for i in top5_indx]
        for i in range(5):
            # Update the value for 'key'
            if top5_class[i] in dict:
                dict[top5_class[i]] += probs.top5conf[i].item()
            else:
                dict[top5_class[i]] = probs.top5conf[i].item()

    # Sort the dictionary items based on values in descending order
    sorted_items = sorted(dict.items(), key=lambda x: x[1], reverse=True)

    # Select the top 5 items
    top5_items = sorted_items[:5]

    # Copy the dictionary
    new_dict = dict.copy()

    # Divide each value by 5 in the original dictionary
    for key, value in top5_items:
        new_dict[key] = value / len(masked_images)

    # Change the file extension to ".json"
    file_name = video_path.rsplit('.', 1)[0] + '.json'

    # Create the file and save the top 5 dictionary to it
    with open(file_name, 'w') as file:
        json.dump(new_dict, file)


def classify_videos(video_path, yolo, num_frames, time_interval):
    predictor = init_sam_model(MODEL_TYPE, DEVICE, SAM_CHECKPOINT)
    for subdir, dirs, files in os.walk(video_path):
        if files:
            # Get the last part of directory which is considered as the video number
            video_number = os.path.basename(subdir)
            # Sort files to ensure naming is in order
            files.sort()

            # Enumerate files with 1-based index and construct name
            for index, file in enumerate(files, start=1):
                # Load the video using OpenCV
                video_path = os.path.join(subdir, file)
                frames = extract_frames(video_path, num_frames, time_interval)
                masked_images = process_image(frames, predictor)
                yolo_predict(video_path, masked_images, yolo)