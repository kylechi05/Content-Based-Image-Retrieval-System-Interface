import os

import cv2
import json
import numpy as np

from histogram_intersection import compute_3d_hist, compute_lbp_hist
from vp_tree import build_vptree, search_vptree, VPNode

'''
Script to find the best similarity radius r for histogram intersection-based search
    - loads precomputed distance matrix from distance_matrix.npy
    - For each similarity radius r in [0, 1] with step size 0.01:
        - for each ground truth cluster from clustering.py
            - for each image in the cluster
                - retrieve the image's row from the distance matrix containing histogram distances to all other images
                - "retrieve"/count all images within distance r
                - compute precision, recall, and F1 score based on "relevant" images defined by the cluster
        - achieves an F1 score for every image in every cluster for radius r
            - i.e. F1 score for every image acting as query image against ground truth defined by its cluster, for radius r
    - averages F1 scores for each radius r
    - outputs optimal radius r based on highest average F1 score using ground truth clusters
'''

# load image features for VP tree
segmented_images = "images/segmented"
segmented_image_files = [os.path.join(segmented_images, f) for f in os.listdir(segmented_images)]
segmented_image_files = sorted(segmented_image_files)

img_to_idx = {img_file: idx for idx, img_file in enumerate(segmented_image_files)}

# load distance matrix and clusters
distance_matrix = np.load("distance_matrix.npy")
with open("clusters/k_means_clusters.json", "r") as f: # adjust name to test other ground truth clusters
    clusters = json.load(f)["clusters"]

# find optimal radius r
best_r = None
best_f1 = 0.0

for r in np.arange(0, 1, 0.01):
    r_f1_scores = []
    r_precision = []
    r_recall = []

    for cluster in clusters:
        relevant_images = set(clusters[cluster])
        
        for img in clusters[cluster]:
            # get features for the query image
            query_feature = img_to_idx[img]
            
            # distance matrix
            retrieved_indices = np.where(distance_matrix[query_feature] <= r)[0]
            retrieved_images = {segmented_image_files[i] for i in retrieved_indices}

            true_positives = len(relevant_images.intersection(set(retrieved_images)))
            false_positives = len(retrieved_images) - true_positives
            false_negatives = len(relevant_images) - true_positives
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            r_precision.append(precision)
            r_recall.append(recall)
            r_f1_scores.append(f1_score)

    avg_precision = sum(r_precision) / len(r_precision) if r_precision else 0
    avg_recall = sum(r_recall) / len(r_recall) if r_recall else 0
    avg_f1 = sum(r_f1_scores) / len(r_f1_scores) if r_f1_scores else 0
    print(f"Radius: {r:.2f}, Average F1 Score: {avg_f1:.4f} Average Precision: {avg_precision:.4f}, Average Recall: {avg_recall:.4f}")
    
    if avg_f1 > best_f1:
        best_f1 = avg_f1
        best_r = r

print(best_r, best_f1)