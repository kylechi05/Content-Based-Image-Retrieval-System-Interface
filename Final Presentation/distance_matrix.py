import os

import cv2
import numpy as np

from histogram_intersection import compute_3d_hist, compute_lbp_hist, get_histogram_distance

'''
Script to compute the distance matrix for all segmented parasite images
    - loads segmented parasite images from "images/segmented"
    - computes color and texture histograms for each image
    - computes histogram intersection distance between each pair of images
    - outputs distance matrix as a 2D list
'''

segmented_images = "images/segmented"
segmented_image_files = [os.path.join(segmented_images, f) for f in os.listdir(segmented_images)]
segmented_image_files = sorted(segmented_image_files)

image_features = []
for img_file in segmented_image_files:
    image = cv2.imread(img_file)
    color_hist = compute_3d_hist(image)
    lbp_hist = compute_lbp_hist(image)
    image_features.append((color_hist, lbp_hist))

distance_matrix = [[0 for _ in range(len(segmented_image_files))] for _ in range(len(segmented_image_files))]

for i in range(len(segmented_image_files)):
    for j in range(i, len(segmented_image_files)):
        color_hist1, lbp_hist1 = image_features[i]
        color_hist2, lbp_hist2 = image_features[j]
        distance = get_histogram_distance(color_hist1, color_hist2, lbp_hist1, lbp_hist2, a=0.2, b=0.8)
        distance_matrix[i][j] = distance
        distance_matrix[j][i] = distance

# save distance matrix to np file
distance_matrix_np = np.array(distance_matrix)
print(distance_matrix_np.max(), distance_matrix_np.min())
np.save("distance_matrix.npy", distance_matrix_np)
        