from frame_extract import extract_frame
from segmentation import segment_images
from predict import yolo_train
from video_inf import classify_videos

# Path to the parent directory
VIDEOS_PATH = '/home/mine01/Desktop/code/AWP/Cows_identification/videos'
TEST_VIDEOS_PATH = '/home/mine01/Desktop/code/AWP/Cows_identification/test_videos'
FRAMES_PATH = '/home/mine01/Desktop/code/AWP/Cows_identification/frames'

# Global variables for frame extraction
NUM_FRAMES = 25  # number of frames to extract from each video
TIME_INTERVAL = 0.1  # time interval in seconds between each extracted frame

TEST_NUM_FRAMES = 30
TEST_TIME_INTERVAL = 0.1


if __name__ == "__main__":
    # extract_frame(VIDEOS_PATH, FRAMES_PATH, NUM_FRAMES, TIME_INTERVAL)
    # segment_images(FRAMES_PATH)
    yolo = yolo_train(train=False)
    classify_videos(TEST_VIDEOS_PATH, yolo, TEST_NUM_FRAMES, TEST_TIME_INTERVAL)


