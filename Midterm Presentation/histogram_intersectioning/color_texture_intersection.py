import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage.feature import local_binary_pattern

query_image = cv2.imread('images/milkyway2.png')
database_images = {
    "Milky Way": cv2.imread('images/milkyway1.png'),
    "Aurora": cv2.imread('images/aurora1.png'),
    "Ocean": cv2.imread('images/ocean1.png'),
    "Sunset": cv2.imread('images/sunset1.png')
}

r = 3
n_points = 8 * r
METHOD = 'uniform'

def compute_3d_hist(image, bins=(8,8,8)):
    hist = cv2.calcHist([image], [0,1,2], None, bins, [0,256,0,256,0,256])
    hist = hist / hist.sum()
    return hist

def compute_lbp_hist(gray_image, n_points, radius, method, n_bins):
    lbp = local_binary_pattern(gray_image, n_points, radius, method)
    hist = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))[0]
    hist = hist / hist.sum()
    return hist

def histogram_intersection(hist1, hist2):
    return np.minimum(hist1, hist2).sum()

query_color_hist = compute_3d_hist(query_image)
query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
query_lbp_hist = compute_lbp_hist(query_gray, n_points, r, METHOD, n_points + 2)

scores = {}
for name, img in database_images.items():
    db_color_hist = compute_3d_hist(img)
    color_score = histogram_intersection(query_color_hist, db_color_hist)
    
    db_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    db_lbp_hist = compute_lbp_hist(db_gray, n_points, r, METHOD, n_points + 2)
    texture_score = histogram_intersection(query_lbp_hist, db_lbp_hist)
    
    combined_score = (color_score + texture_score) / 2
    scores[name] = combined_score

n_db = len(database_images)
fig = plt.figure(figsize=(15, 6))
gs = GridSpec(2, n_db, height_ratios=[1, 1.2])

ax_query = fig.add_subplot(gs[0, n_db//2])
ax_query.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
ax_query.set_title("Query Image")
ax_query.axis("off")

for i, (name, img) in enumerate(database_images.items()):
    ax_db = fig.add_subplot(gs[1, i])
    ax_db.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_db.set_title(f"{name}\nScore: {scores[name]:.3f}")
    ax_db.axis("off")

plt.tight_layout()
plt.show()