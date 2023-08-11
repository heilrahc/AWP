# Cows identification

This is a research project that trying to identify a cow's id with its side view

## Prerequisites

pre-requisite is summarized in requirements.txt

pip install -r requirements.txt

### Code Structure:
- `main.py`: where you initialize the code, you can find all the arguments there which could guide you on how to run the program
- `frame_extract.py`: code for extracting frames from a videos
- `segementation.py`: code for segmenting an object(specified by COCO object indx) from a frame, in the main use case it will be cattle. 
- `predict.py`: code for training a yolo classification model
- `video_inf_cls.py`: code for classifying on testing/unknown videos, once you have a trained model

### Getting Started

1. To train a model, first you need to have a folder of videos for training. The folder should be as the following format. The folder for testing should be of the same format:

   - `video_folder/`: The main directory.
       - `class1/`: The dir containing all videos of class 1.
         - `video`
         - `video`
       - `class2/`: The dir containing all videos of class 2.
         - `video`
         - `video`
       - `class3/`: The dir containing all videos of class 3.
         - `video`
         - `video`
   
    Note: there are some code under utils/ that might help you build a dataset.

2. Then you can start training by the following command

    python3 main.py --train_videos data/train_videos --inf_videos data/test_videos --cls_model m --seg_model m --epochs 30

    the above command trains a yolo medium model by 30 epochs and do inference on test videos. The results will be printed
    in the console and saved under the same folder as the test videos. 
    
    Details arguments you can add is listed in main.py.

3. If you have dataset built already,  you can replace the --train_videos to --train_dataset to skip the frame extraction and segmentation

4. If you have a trained classification model and want to re-train it you can just add arguments: --trained_cls_model and set --retrain to true;
   If you don't want to retrain it, then set --retrain to false and you don't need to specify a dataset for training.