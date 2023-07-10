import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil


def extract_frames(path, frames_path, num_frames, time_interval):
    # Path to the parent directory
    path = path
    frames_path = frames_path

    # Make sure the frame directory exists
    os.makedirs(frames_path, exist_ok=True)

    # List to store video names
    video_names = []

    # Global variables for frame extraction
    NUM_FRAMES = num_frames  # number of frames to extract from each video
    TIME_INTERVAL = time_interval  # time interval in seconds between each extracted frame

    # Loop through all directories and subdirectories
    for subdir, dirs, files in os.walk(path):
        # Check if directory is not empty
        if files:
            # Get the last part of directory which is considered as the video number
            video_number = os.path.basename(subdir)
            # Sort files to ensure naming is in order
            files.sort()
            # Enumerate files with 1-based index and construct name
            for index, file in enumerate(files, start=1):
                video_name = f"{video_number}_{index}_mp4"
                video_names.append(video_name)

                # Load the video using OpenCV
                video_path = os.path.join(subdir, file)
                vidcap = cv2.VideoCapture(video_path)

                # Get video frames per second (fps)
                fps = vidcap.get(cv2.CAP_PROP_FPS)
                # Calculate the frame skip based on desired time interval
                frame_skip = int(fps * TIME_INTERVAL)

                # Get total frames
                total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
                # Calculate the start and end frame for the middle few 30%
                start_frame = int(total_frames * 0.35)
                end_frame = int(total_frames * 0.65)

                success, image = vidcap.read()
                frame_count = 0
                extracted_frames = 0
                while success:
                    # Check if this frame is one of the frames we want to extract
                    if frame_count % frame_skip == 0 and extracted_frames < NUM_FRAMES and start_frame <= frame_count <= end_frame:
                        frame_name = f"{video_name}_{extracted_frames + 1}.png"
                        frame_path = os.path.join(frames_path, frame_name)
                        cv2.imwrite(frame_path, image)

                        extracted_frames += 1
                    success, image = vidcap.read()
                    frame_count += 1


def extract_frame(video_path, frames_path, num_frames, time_interval):
    extract_frames(video_path, frames_path, num_frames, time_interval)

