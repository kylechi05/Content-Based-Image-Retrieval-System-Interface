from skimage.feature import local_binary_pattern
import numpy as np
import cv2
import matplotlib.pyplot as plt

image1 = cv2.imread('images/aurora1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.flip(image1, -1)

radius = 3
n_points = 8 * radius 
METHOD = 'uniform'

lbp1 = local_binary_pattern(image1, n_points, radius, METHOD)
lbp2 = local_binary_pattern(image2, n_points, radius, METHOD)


n_bins = n_points + 2

hist1 = np.histogram(lbp1.ravel(), bins=n_bins, range=(0, n_bins))[0]
hist2 = np.histogram(lbp2.ravel(), bins=n_bins, range=(0, n_bins))[0]

hist1 = hist1 / hist1.sum()
hist2 = hist2 / hist2.sum()

intersection = np.minimum(hist1, hist2)
intersection_score = intersection.sum()
print(f'LBP Intersection Score: {intersection_score:.3f}')

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title("Flipped Image")
plt.axis("off")

plt.subplot(2, 1, 2)
plt.plot(hist1, color='black', label='Image 1')
plt.plot(hist2, color='gray', label='Image 2')
plt.fill_between(range(n_bins), intersection, color='blue', alpha=0.3,
                 label=f'Intersection = {intersection_score:.3f}')
plt.xlabel("LBP Code")
plt.ylabel("Normalized Frequency")
plt.title("uDRLBP Histogram Overlap")
plt.legend()
plt.show()
