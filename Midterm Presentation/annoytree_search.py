import os
import time
import cv2

import numpy as np
from annoy import AnnoyIndex
from lib.histogram_intersection import get_hist, histogram_distance

def main():    
    image_dir = 'images'
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = [cv2.imread(f) for f in image_files]
    image_features = []

    for img in images:
        feature_vector = get_hist(img)
        image_features.append(feature_vector)

    feature_dim = len(image_features[0])
    relevant_images = {'images\\1.png', 'images\\2.png', 'images\\3.png', 'images\\4.png',
                       'images\\5.png', 'images\\6.png', 'images\\7.png', 'images\\8.png',
                       'images\\9.png', 'images\\10.png'}
    num_relevant = len(relevant_images)

    simulations = 100
    total_relevant_retrieved = 0
    total_retrieved = 0
    total_time = 0
    fetched_relevant_images = set()

    annoy_index = AnnoyIndex(feature_dim, metric="manhattan")
    for i, feat in enumerate(image_features):
        annoy_index.add_item(i, feat)
    n_trees = 10
    annoy_index.build(n_trees)

    for i in range(simulations):
        query_parasite_image = np.random.randint(1, 11)
        query_image_path = f'images/{query_parasite_image}.png'

        print('################################')
        print(f'Simulation {i+1}')
        print('################################')
        print(f'Annoy initilized with {len(image_features)} images')
        print(f'Querying Annoy with image: {query_image_path}')
        print('================================')

        query_img = cv2.imread(query_image_path)
        query_feature = get_hist(query_img).flatten()

        k=10
        start_time = time.perf_counter()
        nearest_indices = annoy_index.get_nns_by_vector(query_feature, n=k, search_k=-1, include_distances=True)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        retrieved_relevant = 0
        retrieved = len(nearest_indices[0])

        for idx, dist in zip(*nearest_indices):
            img_name = image_files[idx]
            dist = histogram_distance(image_features[idx], query_feature)
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

if __name__ == "__main__":
    main()