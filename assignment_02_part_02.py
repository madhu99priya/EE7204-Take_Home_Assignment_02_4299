import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def region_growing(image, seed, threshold):
    height, width = image.shape
    visited = np.zeros_like(image, dtype=bool)
    segmented = np.zeros_like(image, dtype=np.uint8)

    seed_value = int(image[seed])
    queue = deque([seed])

    while queue:
        x, y = queue.popleft()
        if visited[x, y]:
            continue

        visited[x, y] = True
        pixel_value = int(image[x, y])

        if abs(pixel_value - seed_value) <= threshold:
            segmented[x, y] = 255  

            # using the 4-connected neighbors
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
                    queue.append((nx, ny))

    return segmented


image_path = './input_image.jpg' 

# Read image and convert to grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError(f"Cannot read image at {image_path}")

# Picking a seed point 
seed_point =  (250,250)

# Picking a threshold value
threshold = 38

# Applying the Region Growing technique
segmented = region_growing(image, seed_point, threshold)

# Show results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Original Grayscale Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Region Growing Segmentation")
plt.imshow(segmented, cmap='gray')

plt.tight_layout()
plt.show()

