import os

import cv2
import json
import numpy as np

from histogram_intersection import compute_3d_hist, compute_lbp_hist
from vp_tree import build_vptree, search_vptree, VPNode

'''
Script to evaluate the histogram intersection search against ground truth clusters using best radius r
    - best radius r deteremined in find_similarity.py
    - use the VP-Tree implementation from vp_tree.py
    - averages recall, precision, and F1 scores based on "relevant" images defined by different clusters from clustering.py
'''

# load image features for VP tree
segmented_images = "images/segmented"
segmented_image_files = [os.path.join(segmented_images, f) for f in os.listdir(segmented_images)]
segmented_image_files = sorted(segmented_image_files)

image_features = []
for img_file in segmented_image_files:
    image = cv2.imread(img_file)
    color_hist = compute_3d_hist(image)
    lbp_hist = compute_lbp_hist(image)
    image_features.append((img_file, (color_hist, lbp_hist)))

img_to_idx = {img_file: idx for idx, img_file in enumerate(segmented_image_files)}

# build VP tree
vptree_root = build_vptree(image_features)

# get clusters
cluster_dir = "clusters"
dif_clusters = []

for cluster_file in os.listdir(cluster_dir):
    path = os.path.join(cluster_dir, cluster_file)

    name = cluster_file.split(".")[0]
    with open(path, "r") as f:
        clusters = json.load(f)["clusters"]
        dif_clusters.append((name, clusters))

# evaluate at best radius r found previously
r = 0.14

# loops over different clustering methods
for method_name, clusters in dif_clusters:
    r_f1_scores = []
    r_precision = []
    r_recall = []

    # nested for loop here just iterates over all images
    for cluster in clusters:
        relevant_images = set(clusters[cluster])
        
        for img in clusters[cluster]:
            # get features for the query image
            query_feature_idx = img_to_idx[img]
            query_feature = image_features[query_feature_idx][1]
            
            # search VP tree
            retrieved_images, _ = search_vptree(vptree_root, query_feature, tau=r)
            retrieved_image_names = [name for name, _ in retrieved_images]

            true_positives = len(relevant_images.intersection(set(retrieved_image_names)))
            false_positives = len(retrieved_image_names) - true_positives
            false_negatives = len(relevant_images) - true_positives
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            r_precision.append(precision)
            r_recall.append(recall)
            r_f1_scores.append(f1_score)

    avg_recall = sum(r_recall) / len(r_recall) if r_recall else 0
    avg_precision = sum(r_precision) / len(r_precision) if r_precision else 0
    avg_f1 = sum(r_f1_scores) / len(r_f1_scores) if r_f1_scores else 0
        
    print(f"Method: {method_name} at radius r={r:.2f}: Avg Recall={avg_recall:.4f}, Avg Precision={avg_precision:.4f}, Avg F1 Score={avg_f1:.4f}")