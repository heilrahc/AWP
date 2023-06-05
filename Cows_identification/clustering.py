from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.models import Model
from ultralytics import YOLO

# for loading/processing the images
from tensorflow.keras.utils import load_img

# clustering and dimension reduction
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors


yolo = YOLO('runs/classify/train7/weights/best.pt')
test_path = "/home/mine01/Desktop/code/AWP/SAM_Clustering/cows_datasets/test"

# Define the generators for training and testing data
datagen = ImageDataGenerator(rescale = 1./255,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             horizontal_flip = True)

test_generator = datagen.flow_from_directory(test_path,
                                             target_size = (640, 640),
                                             batch_size = 16,
                                             class_mode = 'categorical')


# define a new model that outputs features from a specific layer
feature_extractor = Model(inputs=yolo.inputs, outputs=yolo.get_layer('layer_name').output)

# get features
features = feature_extractor.predict(test_generator)

def view_cluster(cluster, kmeans):
    # Get filenames from the generator
    filenames = test_generator.filenames

    # Get the indices of files in this cluster
    indices = np.where(kmeans.labels_ == cluster)[0]

    # # Limit to 30 images
    # if len(indices) > 30:
    #     print(f"Clipping cluster size from {len(indices)} to 30")
    #     indices = indices[:30]

    plt.figure(figsize = (15, 10))

    # Loop over the images of this cluster
    for i, index in enumerate(indices):
        img = load_img(os.path.join(test_path, filenames[index]))
        plt.subplot(5, 6, i + 1)
        plt.imshow(img)
        plt.axis('off')

    plt.show()

# Reduce dimensionality
pca = PCA(n_components=12, random_state=22)
pca.fit(features)
features_pca = pca.transform(features)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=19, random_state=22)
kmeans.fit(features_pca)

# Loop through the clusters
for i in range(kmeans.n_clusters):
    print(f"Cluster {i}:")
    view_cluster(i, KMeans)
