import os
import cv2
from collections import defaultdict
import numpy as np
import json
from ultralytics import YOLO
import torch
import torch.nn as nn
import shutil
from sklearn.neighbors import NearestNeighbors
from shutil import move
import numpy as np


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


def yolo_predict(video_path, masked_images, cls_model):
    all_vectors = []
    # Get a list of all layers in the model
    layers = list(cls_model.model.children())

    # Create a new model that includes all layers up to the second to last one
    feature_model = nn.Sequential(*layers[:-1])

    for idx, image in enumerate(masked_images):
        result = feature_model(image)
        all_vectors.append(result)

    # Concatenate all vectors into a large matrix
    all_vectors = np.stack(all_vectors)

    # Compute the mean along the 0-th axis (which corresponds to the "frame" axis)
    mean_vector = np.mean(all_vectors, axis=0)

    return mean_vector


def extract_features_for_video(video_path, cls_model, seg_model, num_frames, time_interval):
    frames = extract_frames(video_path, num_frames, time_interval)
    masked_images = process_image(frames, seg_model)
    feature_vector = yolo_predict(video_path, masked_images, cls_model)

    # Here you need to get the feature vector from top5 or cls_model
    # You might need to modify your cls_model or yolo_predict function to return feature vector
    # Let's assume it's top5 for now

    return np.array(feature_vector)


def build_knn_model(labelled_videos_path, cls_model, seg_model, num_frames, time_interval):
    X = []  # To store feature vectors
    Y = []  # To store labels

    for subdir, dirs, files in os.walk(labelled_videos_path):
        if files:
            label = os.path.basename(subdir)  # Assume subdir is the label
            for file in files:
                if file.endswith(('.MP4')):
                    video_path = os.path.join(subdir, file)
                    feature_vector = extract_features_for_video(video_path, cls_model, seg_model, num_frames,
                                                                time_interval)
                    print(feature_vector.shape)
                    X.append(feature_vector)
                    Y.append(label)

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(X, Y)

    return knn


def cluster_unlabelled_videos(unlabelled_videos_path, labelled_videos_path, cls_model, seg_model, num_frames,
                              time_interval):
    # Build KNN model
    knn = build_knn_model(labelled_videos_path, cls_model, seg_model, num_frames, time_interval)

    # Predict and move unlabelled videos
    for video_file in os.listdir(unlabelled_videos_path):
        video_path = os.path.join(unlabelled_videos_path, video_file)
        feature_vector = extract_features_for_video(video_path, cls_model, seg_model, num_frames, time_interval)

        # Predict label for unlabelled video
        label = knn.predict([feature_vector])[0]

        # Move video to the predicted subfolder in labelled directory
        destination_folder = os.path.join(labelled_videos_path, label)
        move(video_path, destination_folder)

