import os

import cv2
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import linkage
from matplotlib import pyplot as plt
import json
from collections import defaultdict

from histogram_intersection import compute_3d_hist, compute_lbp_hist

'''
Clustering script to cluster similar parasites for ground truth
    - combines color and texture histograms into single feature vector for clustering
    - feature vectors weight color and texture components by a and b respectively
    - attempt different clustering methods and different distance measures
        - k-means with euclidean distance, elbow method and human analysis for ground truth clusters
            - resulted in reasonable clusters
        - agglomerative clustering with different distance measures, automatically thresholded using dendrograms
            - euclidean: did not result in good clusters
            - cosine: resulted in better clusters, but not as good as k-means
        - DSCAN clustering

    - clusters are then used as ground truth for histogram intersection-based search
'''

# Concatenates histograms to a single feature vector - used for clustering
# Weights color and texture histograms by a and b respectively
def combine_features(color_hist, lbp_hist, a=0.1, b=0.9):
    color_hist = color_hist * a
    lbp_hist = lbp_hist * b
    return np.concatenate([color_hist, lbp_hist])


# load images
image_dir = "images/segmented"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

# compute features for each image
features = []
filenames = []
for img_file in image_files:
    # get histograms of image
    image = cv2.imread(img_file)
    color_hist = compute_3d_hist(image)
    lbp_hist = compute_lbp_hist(image)

    # concatenate into one feature vector
    feature_vector = combine_features(color_hist, lbp_hist, a=0.2, b=0.8)
    features.append(feature_vector)
    filenames.append(img_file)

# K-means clustering
def kmeans_clustering(features, num_clusters):
    # use elbow method to determine optimal k
    within_cluster_sum_squares = []
    k_range = range(1, 25)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(features)
        within_cluster_sum_squares.append(kmeans.inertia_)

    #Plot elbow curve
    '''
    plt.plot(k_range, within_cluster_sum_squares, 'o-')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Inertia (WSS)')
    plt.title('Elbow Method for Optimal k')
    plt.show()
    '''
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(features)
    return labels

# agglomerative clustering
def agglomerative_clustering(features, method="average", metric="euclidean"):
    # automatically determine threshold from dendrogram
    linkage_matrix = linkage(features, method, metric)
    merge_heights = linkage_matrix[:, 2]
    jump_heights = np.diff(merge_heights)
    jump_idx = np.argmax(jump_heights) + 1
    threshold = merge_heights[jump_idx]
    agglom = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric=metric,
        linkage=method
    )


    # manual inspection of predetermined clusters
    agglom = AgglomerativeClustering(
        n_clusters=12,
        distance_threshold=None,
        metric=metric,
        linkage=method
    )


    labels = agglom.fit_predict(features)
    return labels

# DBSCAN clustering
def dbscan_clustering(features, eps=0.5, min_samples=5, metric="euclidean"):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = dbscan.fit_predict(features)
    return labels

# kmeans clustering labels
labels = kmeans_clustering(features, num_clusters=12)

# agglomerative clustering labels
labels = agglomerative_clustering(features, method="average", metric="cosine")

# dbscan clustering labels
# labels = dbscan_clustering(features, eps=0.15, min_samples=1, metric="manhattan")

# adjust the above comments, num_clusters, etc functions to produce and test different clustering methods
clusters = defaultdict(list)
for img_file, label in zip(filenames, labels):
    clusters[int(label)].append(img_file)

for cluster_id, imgs in clusters.items():
    print(f"Cluster {cluster_id}: {len(imgs)} images")

output_file = "clusters/clusters.json"
with open(output_file, 'w') as f:
    json.dump({"clusters": clusters}, f, indent=4)
