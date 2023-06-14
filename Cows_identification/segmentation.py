import sys
from segment_anything import sam_model_registry, SamPredictor
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

sys.path.append("..")
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

train_path = '/home/mine01/Desktop/code/AWP/Cows_identification/cows_datasets_2/train'
test_path = '/AWP/Cows_identification/cows_datasets_2/test'
frame_dir = '/home/mine01/Desktop/code/AWP/Cows_identification/frames2'

# Make sure the frame directory exists
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# The dictionary to store video assignments
video_dict = defaultdict(str)

def process_image(image_path, train_path, test_path, test_ratio=1.0):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)
    input_point = np.array([[600, 500], [630, 530], [570, 470]])
    input_label = np.array([1, 1, 1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # show_mask(masks, plt.gca())
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('off')
    # plt.show()

    # Apply the mask to the image
    masked_image = image * masks[0][:, :, None]  # If masks has more than 1 dimension, select the relevant one

    # Convert the masked image back to BGR color scheme for saving
    masked_image = cv2.cvtColor(masked_image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Extract cow id and video index from file name
    cow_id, video_id, _, _ = os.path.basename(image_path).split('_')

    # Check if this video has been assigned before
    if not video_dict[f"{cow_id}_{video_id}"]:
        # If not, assign it now
        if np.random.rand() < test_ratio:
            video_dict[f"{cow_id}_{video_id}"] = test_path
        else:
            video_dict[f"{cow_id}_{video_id}"] = train_path

    # Get the destination path for this video
    new_dir = os.path.join(video_dict[f"{cow_id}_{video_id}"], cow_id)

    # If the destination subdirectory doesn't exist, create it
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Save the masked image to specific subdirectory
    cv2.imwrite(os.path.join(new_dir, f'mask_{os.path.basename(image_path)}'), masked_image)


# Process each image in the frame directory
for image_file in os.listdir(frame_dir):
    image_path = os.path.join(frame_dir, image_file)
    print(image_path)
    if os.path.isfile(image_path):
        process_image(image_path, train_path, test_path)