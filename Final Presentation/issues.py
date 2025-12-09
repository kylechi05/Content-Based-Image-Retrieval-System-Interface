import os

import skimage as ski
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import regionprops
from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt

'''
Demonstrating some fundamental issues I encountered during my project
'''

img_file_1 = "images/issues/image1.png"
img_file_2 = "images/issues/image2.png"

def get_measures(image_file):
    parasites = ski.io.imread(image_file, as_gray=True)

    # convert to grayscale [0,255]

    parasites = (parasites * 255).astype(np.uint8)

    parasites = 255 - parasites


    # apply Otsu threshold
    thresh = threshold_otsu(parasites)
    binary = parasites > thresh

    # distance transform
    distance = ndi.distance_transform_edt(binary)

    # Gaussian smoothing
    distance_smooth = gaussian(distance, sigma=1)

    # find markers for each parasite object
    coords = peak_local_max(distance_smooth, labels=binary, min_distance=40)
    markers = np.zeros_like(distance_smooth, dtype=int)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i
    
    # apply watershed for labels
    labels = watershed(-distance, markers, mask=binary)

    return distance_smooth, labels

p1_distance_smooth, p1_labels = get_measures(img_file_1)
p2_distance_smooth, p2_labels = get_measures(img_file_2)

fig, axes = plt.subplots(3, 2, figsize=(8,10))

raw1 = ski.io.imread(img_file_1)
raw2 = ski.io.imread(img_file_2)

axes[0,0].imshow(raw1)
axes[0,0].set_title("Raw Image 1")
axes[0,0].axis('off')

axes[0,1].imshow(raw2)
axes[0,1].set_title("Raw Image 2")
axes[0,1].axis('off')

axes[1,0].imshow(p1_distance_smooth, cmap='jet')
axes[1,0].set_title("Distance Transform Smoothed 1")
axes[1,0].axis('off')

axes[1,1].imshow(p2_distance_smooth, cmap='jet')
axes[1,1].set_title("Distance Transform Smoothed 2")
axes[1,1].axis('off')

axes[2,0].imshow(p1_labels, cmap='nipy_spectral')
axes[2,0].set_title("Watershed Labels 1")
axes[2,0].axis('off')

axes[2,1].imshow(p2_labels, cmap='nipy_spectral')
axes[2,1].set_title("Watershed Labels 2")
axes[2,1].axis('off')

plt.show()

