import os
import time

import cv2
import numpy as np
from lib.histogram_intersection import get_hist, histogram_distance

class VPNode:
    def __init__(self, image_name=None, pivot=None, mu=None, left=None, right=None, points=None):
        self.image_name = image_name
        self.pivot = pivot
        self.mu = mu
        self.left = left
        self.right = right
        self.points = points

def build_vptree(images_features):
    if len(images_features) == 0:
        return None
    if len(images_features) == 1:
        name, feature = images_features[0]
        return VPNode(image_name=name, pivot=feature, points=[name])
        
    pivot_idx = np.random.randint(len(images_features))
    pivot_name, pivot_feature = images_features[pivot_idx]

    distances = []
    for i, (name, feature) in enumerate(images_features):
        if i != pivot_idx:
            dist = histogram_distance(pivot_feature, feature)
            distances.append((dist, name, feature))
    
    dists_only = [d[0] for d in distances]
    mu = np.median(dists_only)
    
    left_points = [(name, feat) for d, name, feat in distances if d <= mu]
    right_points = [(name, feat) for d, name, feat in distances if d > mu]
    
    left_tree = build_vptree(left_points)
    right_tree = build_vptree(right_points)
    
    return VPNode(image_name=pivot_name,
                  pivot=pivot_feature,
                  mu=mu,
                  left=left_tree,
                  right=right_tree,
                  points=[name for name, _ in images_features]
                  )

def search_vptree(root, query_feature, tau):
    if root is None:
        return []

    comparisons = 0
    fetched_relevant_images = []
    stack = [root]

    while stack:
        node = stack.pop()
        if node is None:
            continue

        dist = histogram_distance(node.pivot, query_feature)
        comparisons += 1

        if dist <= tau:
            fetched_relevant_images.append((node.image_name, dist))

        if node.left is None and node.right is None:
            continue

        if dist < node.mu:
            stack.append(node.left)
            if dist + tau >= node.mu:
                stack.append(node.right)
        else:
            stack.append(node.right)
            if dist - tau <= node.mu:
                stack.append(node.left)

    return (fetched_relevant_images, comparisons)

def main():    
    image_dir = 'images'
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = [cv2.imread(f) for f in image_files]
    image_features = []

    for img, img_name in zip(images, image_files):
        feature_vector = get_hist(img)
        image_features.append((img_name, feature_vector))

    relevant_images = {'images\\1.png', 'images\\2.png', 'images\\3.png', 'images\\4.png',
                       'images\\5.png', 'images\\6.png', 'images\\7.png', 'images\\8.png',
                       'images\\9.png', 'images\\10.png'}
    num_relevant = len(relevant_images)

    simulations = 100
    total_time = 0
    total_relevant_retrieved = 0
    total_retrieved = 0
    fetched_relevant_images = set()


    for i in range(simulations):
        vp_tree = build_vptree(image_features)

        query_parasite_image = np.random.randint(1, 11)
        query_image_path = f'images/{query_parasite_image}.png'

        print('################################')
        print(f'Simulation {i+1}')
        print('################################')
        print(f'VP Tree initilized with {len(image_features)} images')
        print(f'Root Node: {vp_tree.image_name}')
        print(f'Querying VP tree with image: {query_image_path}')
        print('================================')

        query_img = cv2.imread(query_image_path)
        query_feature = get_hist(query_img)

        tau = 0.25

        start_time = time.perf_counter()
        fetched, _ = search_vptree(vp_tree, query_feature, tau)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        retrieved_relevant = 0
        retrieved = len(fetched)

        fetched.sort(key=lambda x: x[1])
        for img_name, dist in fetched:
            print(f'{img_name}: {dist:.4f}')
            fetched_relevant_images.add(img_name)
            if img_name in relevant_images:
                retrieved_relevant += 1

        recall = retrieved_relevant / num_relevant
        precision = retrieved_relevant / retrieved

        print(f'Elapsed time: {elapsed_time:.6f} seconds')
        print(f'Recall: {recall:.4f}, Precision: {precision:.4f}')
         
        total_relevant_retrieved += retrieved_relevant
        total_retrieved += retrieved
        total_time += elapsed_time
    
    recall = total_relevant_retrieved / (num_relevant * simulations)
    precision = total_relevant_retrieved / total_retrieved

    print('================================')
    print('All Relevant Images Across Simulations:')
    for img_name in (fetched_relevant_images):
        print(f'{img_name}')
    print(f'Average time over {simulations} simulations: {total_time/simulations} seconds')
    print(f'Overall Recall: {recall:.4f}, Overall Precision: {precision:.4f}')

if __name__ == '__main__':
    main()
