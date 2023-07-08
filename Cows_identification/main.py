from frame_extract import extract_frame
from segmentation import segment_images
from predict import yolo_train
from video_inf import classify_videos
from ultralytics import YOLO
import argparse


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument('--train_videos', metavar='train_videos', type=str,
                        help='path to your train videos path (if you do not have a train datasets, only raw videos)')
    parser.add_argument('--train_dataset', metavar='train_dataset', type=str, default=None,
                        help='path to your train dataset path')
    parser.add_argument('--frame_path', metavar='frame_path', type=str,
                        help='path to where you (want to) store the frames')
    parser.add_argument('--inf_videos', metavar='inf_videos', type=str, default=None,
                        help='path to the videos you want to do inference on')
    parser.add_argument('--cls_model', metavar='cls_model', type=str, choices=['n', 's', 'm', 'l'], default='s',
                        help='choices of a new yolo classification model')
    parser.add_argument('--seg_model', metavar='seg_model', type=str, choices=['n', 's', 'm', 'l'], default='m',
                        help='choices of a new yolo segmentation model')
    parser.add_argument('--trained_cls_model', metavar='trained_cls_model', type=str, default=None,
                        help='path to the weights of your trained cls model weights')
    parser.add_argument('--retrain', metavar='retrain', type=bool, default=False,
                        help='whether to re-train the model or not')
    parser.add_argument('--epochs', metavar='epochs', type=int, default=50,
                        help='numbers of epochs for training')
    parser.add_argument('--num_frames_t', metavar='num_frames_t', type=int, default=15,
                        help='numbers of frames to extract for training')
    parser.add_argument('--time_interval_t', metavar='time_interval_t', type=float, default=0.2,
                        help='time interval between frames for training')
    parser.add_argument('--num_frames_i', metavar='num_frames_i', type=int, default=3,
                        help='numbers of frames to extract for inference')
    parser.add_argument('--time_interval_i', metavar='time_interval_i', type=float, default=0.3,
                        help='time interval between frames for inference')

    # Parse the arguments
    args = parser.parse_args()

    yolo_seg = YOLO("yolov8" + args.seg_model + "-seg.pt")
    yolo_cls = YOLO("yolov8" + args.cls_model + "-cls.pt")

    # if dataset is not built, built training dataset using videos
    if args.train_dataset is None:
        extract_frame(args.train_videos, args.frame_path, args.num_frames_t, args.time_interval_t)
        segment_images(args.frame_path, yolo_seg)

    # train/load the classification model
    if args.trained_cls_model is None:
        yolo = yolo_train(yolo_cls, args.train_dataset, args.epochs)
    else:
        yolo = yolo_train(yolo_cls, args.train_dataset, args.epochs,
                          yolo_cls_trained_path=args.trained_cls_model, train=args.retrain)

    if args.inf_videos is not None:
        # perform videos inference
        classify_videos(args.inf_videos, yolo, yolo_seg, args.num_frames_i, args.time_interval_i,
                        args.cls_model, args.trained_cls_model)
