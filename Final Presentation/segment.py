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
Script to segment images of parasites into individual parasite images
    - uses Otsu thresholding to create binary mask
    - apply distance transform and watershed algorithm to segment individual parasites
    - removes objects that are too small or too large based on size thresholds
        - removes noise and darker background regions
    - saves segmented parasites to "images/segmented" directory
    - output used for clustering and histogram intersection-based search
'''

raw_image_dir = "images/raw"
raw_image_files = [os.path.join(raw_image_dir, f) for f in os.listdir(raw_image_dir)]

# segmenting parasite images to extract individual parasites
for image_file in raw_image_files:
    # load image
    parasites = ski.io.imread(image_file)
    parasites = 255 - parasites

    # convert to grayscale [0,255]
    parasites = ski.color.rgb2gray(parasites)
    parasites = (parasites * 255).astype(np.uint8)

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
    lables = watershed(-distance_smooth, markers, mask=binary)

    # remove small/large objects based on size thresholds
    min_size = 200
    max_size = 7000
    filtered_labels = np.zeros_like(lables)

    for region in regionprops(lables):
        if min_size <= region.area <= max_size:
            filtered_labels[lables == region.label] = region.label

    # apply masks to original image to get individual parasites
    original_img = ski.io.imread(image_file)
    for region in regionprops(filtered_labels):
        # white background
        mask_img = np.ones_like(original_img, dtype=np.uint8) * 255

        # mask
        mask = filtered_labels == region.label
        mask_img[mask] = original_img[mask]

        # crop image
        minr, minc, maxr, maxc = region.bbox
        parasite_cropped_img = mask_img[minr:maxr, minc:maxc]

        # save
        filename = f"parasite_{os.path.basename(image_file).split('.')[0]}_{region.label}.png"
        ski.io.imsave(os.path.join("images/segmented", filename), parasite_cropped_img)