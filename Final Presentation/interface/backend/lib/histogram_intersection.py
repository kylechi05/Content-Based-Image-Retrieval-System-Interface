import os

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt

'''
Functions for computing histograms and histogram intersection
    - all histograms are normalized to sum to 1
    - histogram intersections are computed seperately for 3D color hist and LBP hist
    - final distance is defined as 1 - (a * color_hist_intersection + b * lbp_hist_intersection)
        - a and b are weights for color and texture components respectively
        - a + b = 1
'''

# 3D color histogram
def compute_3d_hist(image, bins=(8,8,8)):
    hist = cv2.calcHist([image], [0,1,2], None, bins, [0,256,0,256,0,256])
    hist = hist / hist.sum()
    hist = hist.flatten()
    return hist

# LPB texture histogram
def compute_lbp_hist(image, n_points=24, radius=3, method="uniform", n_bins=26):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method)
    hist = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))[0]
    hist = hist / hist.sum()
    return hist

# computes histogram intersection
def get_histogram_intersection(hist1, hist2):
    return np.minimum(hist1, hist2).sum()

# compute histogram distance
def get_histogram_distance(color_hist1, color_hist2, lbp_hist1, lbp_hist2, a=0.2, b=0.8):
    intersection = get_histogram_intersection(color_hist1, color_hist2) * a + get_histogram_intersection(lbp_hist1, lbp_hist2) * b
    distance = np.clip(1 - intersection, 0, 1)
    return distance


