import os
import sys
import time

import cv2
import numpy as np
from lib.histogram_intersection import get_hist, histogram_distance

def main(query_image_path):    
    image_dir = 'images'
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = [cv2.imread(f) for f in image_files]

    image_features = []

    for img, img_name in zip(images, image_files):
        feature_vector = get_hist(img)
        image_features.append((img_name, feature_vector))

    tau = 0.25
    relevant_images = set()

    query_img = cv2.imread(query_image_path)
    query_feature = get_hist(query_img)

    print('################################')

    start_time = time.perf_counter()
    for i in range(len(image_features)):
        img_name, img_feature = image_features[i]
        dist = histogram_distance(query_feature, img_feature)
        if dist <= tau:
            relevant_images.add((img_name, dist))
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    print('================================')
    print('All Relevant Images:')
    for img_name, dist in sorted(relevant_images, key=lambda x: x[1]):
        print(f'{img_name}: {dist:.4f}')
    print(f'Total time taken for exhaustive search: {elapsed_time:.6f} seconds')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Please input only one path to query image')
    
    main(sys.argv[1])
