import cv2
import numpy as np
import matplotlib.pyplot as plt

# Creating a blank image
img = np.zeros((200, 200), dtype=np.uint8)

# Drawing a triangle as one object (value: 85)
triangle_cnt = np.array([[50, 150], [100, 50], [150, 150]])
cv2.drawContours(img, [triangle_cnt], 0, 85, -1)

# Drawing a rectangle as the second object (value: 170)
cv2.rectangle(img, (20, 20), (60, 60), 170, -1)

# Adding Gaussian noise
mean = 0
stddev = 10
gaussian_noise = np.random.normal(mean, stddev, img.shape).astype(np.int16)
noisy_img = np.clip(img.astype(np.int16) + gaussian_noise, 0, 255).astype(np.uint8)

# Applying Otsu's thresholding
_, otsu_thresh = cv2.threshold(noisy_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Displaying results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Noisy Image')
plt.imshow(noisy_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Otsu's Threshold")
plt.imshow(otsu_thresh, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
