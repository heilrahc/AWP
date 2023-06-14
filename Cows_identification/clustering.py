import cv2
import numpy as np
import torch
import torch.nn as nn
import os
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from PIL import Image
from ultralytics import YOLO
from sklearn.decomposition import PCA
import shutil

CLUSTER_NUMBER = 19
# The path of your images
image_folder = "/home/mine01/Desktop/code/AWP/Cows_identification/cows_datasets_2/test"

image_paths = []
features = []

# Load the YOLO model
yolo = YOLO('runs/classify/train2/weights/best.pt')

# Get a list of all layers in the model
layers = list(yolo.model.children())

# Create a new model that includes all layers up to the second to last one
feature_model = nn.Sequential(*layers[:-1])

# Process each image
for subdir, dirs, files in os.walk(image_folder):
    for file in files:
        image_path = os.path.join(subdir, file)
        image_paths.append(image_path)
        # Read the image and convert to the size your model expects
        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 640))

        # Convert image to tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)

        # Extract the features
        feature = feature_model(image_tensor)
        # Reshape the tensor to 1D
        feature_1d = feature.view(-1)
        features.append(feature_1d)

# Convert list of tensors to 2D tensor
features_tensor = torch.stack(features)

# Convert the tensor to numpy array
features_array = features_tensor.numpy()

# Create a PCA object
pca = PCA(n_components=200)

# Fit the PCA model and transform your data
reduced_features = pca.fit_transform(features_array)

# # Perform KMeans clustering on the reduced data
# kmeans = KMeans(n_clusters=CLUSTER_NUMBER, random_state=0, verbose=1).fit(reduced_features)

# Perform KMedoids clustering on the reduced data
kmedoids = KMedoids(n_clusters=CLUSTER_NUMBER, random_state=0).fit(reduced_features)

# Make directories for each cluster
os.mkdir('clusters')

for i in range(CLUSTER_NUMBER):  # change this to match the number of clusters
    os.mkdir(f'clusters/cluster_{i}')

# Copy the images to the respective cluster folders
for i, image_path in enumerate(image_paths):
    if os.path.exists(image_path):  # Check if the source file exists
        cluster = kmedoids.labels_[i]
        target_path = f'clusters/cluster_{cluster}/{os.path.basename(image_path)}'
        print(f"Copying file from {image_path} to {target_path}")  # For debugging
        shutil.copy(image_path, target_path)
    else:
        print(f"Source file does not exist: {image_path}")
