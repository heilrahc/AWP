import os
import cv2
import numpy as np
import json
import torch
from PIL import Image

# TEST_VIDEOS_PATH = '/home/mine01/Desktop/code/AWP/Cows_identification/test_videos'

def find_index_of_class(cls, target=19.):
    equals_target = torch.eq(cls, target)
    if torch.any(equals_target):
        return torch.nonzero(equals_target)[0].item()
    else:
        return None


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
    start_frame = int(total_frames * 0.40)
    end_frame = int(total_frames * 0.60)

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


def process_image(frames, seg_model):
    masked_images = []
    for frame in frames:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        seg_result = seg_model(image)[0]
        indx = find_index_of_class(seg_result.boxes.cls)

        if indx is not None:
            seg_mask = seg_result.masks.data[indx].cpu().numpy()

            # Resize the image to match the mask dimensions
            resized_img = cv2.resize(image, (seg_mask.shape[1], seg_mask.shape[0]))

            # Expand the dimensions of the mask to match the number of channels in the resized image
            seg_mask_expanded = np.expand_dims(seg_mask, axis=2)
            seg_mask_expanded = np.tile(seg_mask_expanded, (1, 1, resized_img.shape[2]))

            # Perform the multiplication
            seg_image = seg_mask_expanded * resized_img

            # convert seg1 back to RGB color scheme
            masked_image = cv2.cvtColor(seg_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            masked_image = auto_crop(masked_image)

            masked_images.append(masked_image)

    return masked_images


def yolo_predict(video_path, masked_images, yolo_finetuned):
    dict = {}
    # Change the file extension to ".json"
    json_path = video_path.rsplit('.', 1)[0] + '.json'
    image_path = video_path.rsplit('.', 1)[0] + '.png'
    for idx, image in enumerate(masked_images):
        result = yolo_finetuned(image)
        if idx == 1:
            result_plotted = result[0].plot()
            Image.fromarray(result_plotted).save(image_path)
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

    # Create the file and save the top 5 dictionary to it
    with open(json_path, 'w') as file:
        json.dump(new_dict, file)

    return new_dict


def classify_videos(test_videos_path, yolo_finetune, seg_model, num_frames, time_interval, model_size, trained_model_path=None):
    # # Train the model
    # predictor.train(data='coco128-seg.yaml', epochs=100, imgsz=640)
    if trained_model_path is None:
        result_name = "accuracy_" + model_size + "_" + str(num_frames) + "frames.txt"
    else:
        result_name = "accuracy_" + trained_model_path.split('/')[2] + "_" + str(num_frames) + "frames.txt"

    num_hits = 0
    num_videos = 0

    for subdir, dirs, files in os.walk(test_videos_path):
        if files:
            score = 0
            subdir_len = 0
            # Get the last part of directory which is considered as the video number
            video_number = os.path.basename(subdir)
            # Sort files to ensure naming is in order
            files.sort()

            # Enumerate files with 1-based index and construct name
            for index, file in enumerate(files, start=1):
                if file.endswith(('.MP4')):
                    # Load the video using OpenCV
                    video_path = os.path.join(subdir, file)
                    frames = extract_frames(video_path, num_frames, time_interval)
                    masked_images = process_image(frames, seg_model)
                    top5 = yolo_predict(video_path, masked_images, yolo_finetune)
                    top1_label = next(iter(top5), None)
                    if top1_label is not None:
                        subdir_len += 1
                        if top1_label == video_number:
                            score += 1
            if subdir_len != 0:
                acc = score / subdir_len
            else:
                acc = 0

            with open(os.path.join(subdir, result_name), 'w') as file:
                file.write(str(acc))

            num_hits += score
            num_videos += subdir_len

    accuracy = num_hits / num_videos

    print("The model correctly classify ", num_hits, "videos out of ", num_videos, "videos")
    print("Accuracy (video level): ", accuracy)

    with open(os.path.join(test_videos_path, result_name), 'w') as file:
        file.write(str(accuracy))

