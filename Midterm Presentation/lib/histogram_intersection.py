import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def compute_3d_hist(image, bins=(8,8,8)):
    hist = cv2.calcHist([image], [0,1,2], None, bins, [0,256,0,256,0,256])
    return hist

def compute_lbp_hist(gray_image, n_points, radius, method, n_bins):
    lbp = local_binary_pattern(gray_image, n_points, radius, method)
    hist = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))[0]
    return hist

def get_hist(image):
    color_hist = compute_3d_hist(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_hist = compute_lbp_hist(gray, n_points=24, radius=3, method='uniform', n_bins=26)
    feature_vector = np.concatenate([color_hist.flatten(), lbp_hist])
    feature_vector = feature_vector / feature_vector.sum()
    return feature_vector

def histogram_intersection(hist1, hist2):
    return np.minimum(hist1, hist2).sum()

def histogram_distance(hist1, hist2):
    intersection = histogram_intersection(hist1, hist2)
    return 1 - intersection