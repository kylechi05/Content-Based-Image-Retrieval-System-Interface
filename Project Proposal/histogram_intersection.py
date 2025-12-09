import cv2
import matplotlib.pyplot as plt
import numpy as np

image1 = cv2.imread('images/milkyway1.png')
image2 = cv2.imread('images/milkyway2.png')

# Display original images
plt.figure(figsize=(25,4))
plt.subplot(1, 5, 1)
plt.title('Original Image 1')
plt.axis('off')
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

plt.subplot(1, 5, 2)
plt.title('Original Image 2')
plt.axis('off')
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

colors = ("red", "green", "blue")
sum_scores = 0

for i, color in enumerate(colors):
    hist1 = cv2.calcHist([image1], [i], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [i], None, [256], [0, 256])

    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    
    plt.subplot(1, 5, 3 + i)
    plt.plot(hist1, color=color)
    
    plt.subplot(1, 5, 3 + i)
    plt.plot(hist2, color=color)

    intersection = np.minimum(hist1, hist2)
    intersection_score = intersection.sum()
    sum_scores += intersection_score
    print(intersection_score)
    plt.xlim([0, 256])
    plt.fill_between(range(256), intersection.flatten(), color=color, alpha=0.3,label=f'Intersection = {intersection_score:.3f}')
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Ratio of Pixels")
    plt.title(f'{color.capitalize()} Overlaps')

print(f'Overall Intersection Score: {sum_scores/3:.3f}')

plt.subplots_adjust(left=0.1, right=0.95, top=0.8, bottom=0.2, wspace=0.6, hspace=0.4)
plt.show()
