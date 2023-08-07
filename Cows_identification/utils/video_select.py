import os
import shutil
from random import sample

def copy_selected_files(src, dst, n):
    files = os.listdir(src)
    # If n is larger than the number of files, keep all files
    selected_files = sample(files, min(n, len(files)))
    for file in selected_files:
        shutil.copy(os.path.join(src, file), dst)

def copy_folder_structure_and_files(src_dir, dst_dir, n):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        if os.path.isdir(s):
            if not os.path.exists(d):
                os.makedirs(d)
            if item == 'train':
                for subfolder in os.listdir(s):
                    sub_src = os.path.join(s, subfolder)
                    sub_dst = os.path.join(d, subfolder)
                    if not os.path.exists(sub_dst):
                        os.makedirs(sub_dst)
                    copy_selected_files(sub_src, sub_dst, n)
            elif item == 'test':
                shutil.copytree(s, d, dirs_exist_ok=True)


src_folder = '/home/mine01/Desktop/code/AWP/Cows_identification/data/split_videos'  # update with your source folder
dst_folder = '/home/mine01/Desktop/code/AWP/Cows_identification/data/limit_2videos'  # update with your destination folder
n = 2  # update with your desired number of videos to copy from each subfolder in 'train'

copy_folder_structure_and_files(src_folder, dst_folder, n)
