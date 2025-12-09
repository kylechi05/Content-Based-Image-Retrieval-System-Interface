import cv2
import matplotlib.pyplot as plt

image1 = cv2.imread('images/milkyway1.png')
image2 = cv2.imread('images/milkyway2.png')

# Display original images
plt.figure(figsize=(25,6))
plt.subplot(2, 5, 1)
plt.title('Original Image 1')
plt.axis('off')
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

plt.subplot(2, 5, 6)
plt.title('Original Image 2')
plt.axis('off')
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

colors = ("red", "green", "blue")

for i, color in enumerate(colors):
    hist1 = cv2.calcHist([image1], [i], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [i], None, [256], [0, 256])
    
    plt.subplot(2, 5, 2 + i)
    plt.plot(hist1, color=color)
    plt.xlim([0, 256])
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.title(f'{color.capitalize()} Channel 1')
    
    plt.subplot(2, 5, 7 + i)
    plt.plot(hist2, color=color)
    plt.xlim([0, 256])
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.title(f'{color.capitalize()} Channel 2')

    plt.subplot(2, 5, 5)
    plt.plot(hist1, color=color)
    plt.xlim([0, 256])
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.title(f'Color Histogram 1')
    
    plt.subplot(2, 5, 10)
    plt.plot(hist2, color=color)
    plt.xlim([0, 256])
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.title(f'Color Histogram 2')

plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.6, hspace=0.4)

plt.show()
