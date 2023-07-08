# Cows identification

This is a research project that trying to identify a cow's id with its side view

## Prerequisites

pre-requisite is summarized in requirements.txt

pip install -r requirements.txt

### Getting Started

First need to build dataset for training and testing. The dataset should be as the following format:

- `dataset/`: The main directory.
    - `class1/`: The dir containing all videos of class 1.
      - `video`
      - `video`
    - `class2/`: The dir containing all videos of class 2.
      - `video`
      - `video`
    - `class3/`: The dir containing all videos of class 3.
      - `video`
      - `video`

Then you can start training by the following command

python3 main.py --train_videos data/train_videos --inf_videos data/test_videos --cls_model m --seg_model m --epochs 30

the above command trains a yolo medium model by 30 epochs and do inference on te