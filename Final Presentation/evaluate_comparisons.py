import os

import cv2
import json
import numpy as np
import time

from histogram_intersection import compute_3d_hist, compute_lbp_hist, get_histogram_distance
from vp_tree import build_vptree, search_vptree, VPNode

'''
Script to evaluate the speed/comparisons of histogram intersection search compared to exhaustive search using best radius r
    - best radius r deteremined in find_similarity.py
    - use the VP-Tree implementation from vp_tree.py
    - runs a large number of simulations with randomly generated VP trees
    - averages number of comparisons made during VP tree search using every image as query image
    - averages time per query using VP tree search with histogram intersection
'''

# load image features for VP tree
segmented_images = "images/segmented"
segmented_image_files = [os.path.join(segmented_images, f) for f in os.listdir(segmented_images)]

image_features = []
for img_file in segmented_image_files:
    image = cv2.imread(img_file)
    color_hist = compute_3d_hist(image)
    lbp_hist = compute_lbp_hist(image)
    image_features.append((img_file, (color_hist, lbp_hist)))

img_to_idx = {img_file: idx for idx, img_file in enumerate(segmented_image_files)}

# get clusters
cluster_dir = "clusters/k_means_clusters.json"
with open(cluster_dir, "r") as f:
    clusters = json.load(f)["clusters"]

# evaluate at best radius r found previously
r = 0.14

# find avg comparisons used in VP tree
total_comparisons = []
total_times = []

# runs simulations
for i in range(5):
    # randomly build VP tree
    shuffled = image_features.copy()
    np.random.shuffle(shuffled)
    vptree_root = build_vptree(shuffled)

    sim_comparisons = []
    sim_times = []
    
    # nested for loop here just iterates over all images
    for cluster in clusters:
        relevant_images = set(clusters[cluster])
        
        for img in clusters[cluster]:
            # get features for the query image
            query_feature_idx = img_to_idx[img]
            query_feature = image_features[query_feature_idx][1]

            # search VP tree
            start_time = time.perf_counter()
            _, comparisons = search_vptree(vptree_root, query_feature, tau=r)
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time
            sim_comparisons.append(comparisons)
            sim_times.append(elapsed_time)


    avg_sim_comparisons = np.mean(sim_comparisons)
    avg_sim_time = np.mean(sim_times)
    print(f"Simulation {i+1}: Average Comparisons = {avg_sim_comparisons:.4f}, Average Time = {avg_sim_time:.4f}s")

    total_comparisons.extend(sim_comparisons)
    total_times.extend(sim_times)

exhaustive_times = []
for img_file in segmented_image_files:
    query_feature_idx = img_to_idx[img_file]
    query_feature = image_features[query_feature_idx][1]
    relevant_images = set()

    start_time = time.perf_counter()
    for i in range(len(image_features)):
        img_name, img_feature = image_features[i][1]
        dist = get_histogram_distance(query_feature[0], img_feature[0], query_feature[1], img_feature[1], a=0.2, b=0.8)
        if dist <= r:
            relevant_images.add((img_name, dist))
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    exhaustive_times.append(elapsed_time)

avg_total_comparisons = np.mean(total_comparisons)
print(f"Overall Average Comparisons per query across all simulations: {avg_total_comparisons:.4f}")

total_images = len(segmented_image_files)
print(f"Exhaustive Search Comparisons per query: {total_images}")

comparison_speedup = total_images / avg_total_comparisons if avg_total_comparisons > 0 else 0
print(f"Comparison speedup using VP Tree: {comparison_speedup:.4f}x")

avg_total_times = np.mean(total_times)
print(f"Overall Average Time per query across all simulations: {avg_total_times:.4f}s")

avg_exhaustive_times = np.mean(exhaustive_times)
print(f"Exhaustive Search Time per query: {avg_exhaustive_times:.4f}s")

# replace 1 with actual value of exhaustive search
time_speedup = avg_exhaustive_times / avg_total_times if avg_total_times > 0 else 0
print(f"Time speedup using VP Tree: {time_speedup:.4f}x")

