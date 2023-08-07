import os
import shutil
import numpy as np

# Specify the base data directory, split directory and the train/test split ratio
base_dir = '/home/mine01/Desktop/code/AWP/Cows_identification/data/raw_videos'
split_dir = '/home/mine01/Desktop/code/AWP/Cows_identification/data/split_videos'
train_ratio = 0.5  # Adjust this value to your needs

# Create train and test directories under the split directory
os.makedirs(os.path.join(split_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'test'), exist_ok=True)

# Find classes based on subdirectories of base_dir
classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Process each class
for class_ in classes:
    # Get all subdirectories in class folder
    subdirs = [os.path.join(base_dir, class_, d) for d in os.listdir(os.path.join(base_dir, class_)) if
               os.path.isdir(os.path.join(base_dir, class_, d))]

    video_files = []

    # If 'good' or 'bad/run' folder (regardless of case) is present, proceed
    for subdir in subdirs:
        if subdir.lower().endswith('good') or subdir.lower().endswith('bad/run') \
                or subdir.lower().endswith('bad/slip') or subdir.lower().endswith('bad/stop'):
            for root, dirs, files in os.walk(subdir):
                video_files.extend(
                    [os.path.join(root, f) for f in files if f.endswith('.mp4') or f.endswith('.MP4')])  # Change the extension as needed

    # Ignore the class if there is one or no video
    if len(video_files) <= 1:
        continue

    # Define paths for class in train and test directories
    train_class_dir = os.path.join(split_dir, 'train', class_)
    test_class_dir = os.path.join(split_dir, 'test', class_)

    # Create class directories in train and test if they don't exist
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Randomly split video_files into train and test
    video_files = np.array(video_files)
    np.random.shuffle(video_files)
    train_files, test_files = np.split(video_files, [int(train_ratio * len(video_files))])

    # Move the video files into their respective directories
    for file in train_files:
        shutil.copy(file, train_class_dir)

    for file in test_files:
        shutil.copy(file, test_class_dir)
