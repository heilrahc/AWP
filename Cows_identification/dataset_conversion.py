import os

# change these to match your dataset
root_dir = '/home/mine01/Desktop/code/AWP/Cows_identification/cows_datasets_detect'

# these will contain the names of all the images for which you've created labels
train_image_files = []
val_image_files = []
test_image_files = []

subsets = ['train', 'val', 'test']
image_files_dict = {'train': train_image_files, 'val': val_image_files, 'test': test_image_files}

for subset in subsets:
    subset_dir = os.path.join(root_dir, subset)
    class_dirs = os.listdir(subset_dir)

    for class_id, class_dir_name in enumerate(class_dirs):
        class_dir = os.path.join(subset_dir, class_dir_name)
        for img_file in os.listdir(class_dir):
            img_name, img_ext = os.path.splitext(img_file)
            # only create labels for images
            if img_ext.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            # the label for each image is just: <class_id> 0.5 0.5 1 1
            yolo_label = f"{class_id} 0.5 0.5 1 1\n"
            label_file = os.path.join(class_dir, img_name + '.txt')

            # save the label
            with open(label_file, 'w') as f:
                f.write(yolo_label)

            # add the image file to the corresponding list
            image_files_dict[subset].append(os.path.join(class_dir, img_file))

# save the list of image files
for subset in subsets:
    with open(os.path.join(root_dir, f'{subset}_images.txt'), 'w') as f:
        f.write('\n'.join(image_files_dict[subset]))
