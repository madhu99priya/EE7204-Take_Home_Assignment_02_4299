# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # -----------------------------
# # Part 1: Otsu’s Thresholding
# # -----------------------------

# def generate_image():
#     img = np.zeros((200, 200), dtype=np.uint8)
#     cv2.rectangle(img, (20, 20), (80, 180), 85, -1)   # Object 1
#     cv2.circle(img, (140, 100), 40, 170, -1)          # Object 2
#     return img

# def add_gaussian_noise(img, mean=0, std=15):
#     gauss = np.random.normal(mean, std, img.shape).astype(np.int16)
#     noisy_img = np.clip(img.astype(np.int16) + gauss, 0, 255).astype(np.uint8)
#     return noisy_img

# def apply_otsu_threshold(img):
#     _, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return otsu_thresh

# def part1_otsu_demo():
#     clean_img = generate_image()
#     noisy_img = add_gaussian_noise(clean_img)
#     otsu_result = apply_otsu_threshold(noisy_img)

#     # Show results
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 3, 1)
#     plt.title("Original Image")
#     plt.imshow(clean_img, cmap='gray')

#     plt.subplot(1, 3, 2)
#     plt.title("Noisy Image")
#     plt.imshow(noisy_img, cmap='gray')

#     plt.subplot(1, 3, 3)
#     plt.title("Otsu Result")
#     plt.imshow(otsu_result, cmap='gray')

#     plt.tight_layout()
#     plt.show()


# # -----------------------------------
# # Part 2: Region Growing Segmentation
# # -----------------------------------

# def region_growing(img, seeds, threshold=10):
#     h, w = img.shape
#     segmented = np.zeros_like(img, dtype=np.uint8)
#     visited = np.zeros_like(img, dtype=bool)
#     region_value = 255

#     for seed in seeds:
#         seed_value = img[seed]
#         stack = [seed]

#         while stack:
#             x, y = stack.pop()
#             if visited[x, y]:
#                 continue
#             visited[x, y] = True
#             if abs(int(img[x, y]) - int(seed_value)) <= threshold:
#                 segmented[x, y] = region_value
#                 for dx in [-1, 0, 1]:
#                     for dy in [-1, 0, 1]:
#                         nx, ny = x + dx, y + dy
#                         if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
#                             stack.append((nx, ny))

#     return segmented

# def part2_region_growing_demo():
#     img = generate_image()
#     noisy_img = add_gaussian_noise(img)

#     # Assume user clicks, here we hardcode seeds inside object regions
#     seeds = [(50, 50), (100, 140)]  # Points inside object 1 and 2
#     segmented = region_growing(noisy_img, seeds, threshold=20)

#     plt.figure(figsize=(10, 3))
#     plt.subplot(1, 3, 1)
#     plt.title("Noisy Image")
#     plt.imshow(noisy_img, cmap='gray')

#     plt.subplot(1, 3, 2)
#     plt.title("Region Grown")
#     plt.imshow(segmented, cmap='gray')

#     plt.subplot(1, 3, 3)
#     plt.title("Overlay")
#     plt.imshow(noisy_img, cmap='gray')
#     plt.imshow(segmented, cmap='jet', alpha=0.5)

#     plt.tight_layout()
#     plt.show()


# # ------------------------
# # Run Both Parts
# # ------------------------
# if __name__ == "__main__":
#     print("Running Part 1: Otsu's Thresholding")
#     part1_otsu_demo()

#     print("\nRunning Part 2: Region Growing Segmentation")
#     part2_region_growing_demo()


import cv2
import numpy as np

# Load grayscale image
img = cv2.imread('./OIP.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found. Make sure the path is correct.")

# Clone image for display
display_img = img.copy()
clicked = False
seed_point = None
threshold = 10  

def region_growing(image, seed_point, threshold):
    h, w = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=bool)

    seed_value = int(image[seed_point])
    stack = [seed_point]

    while stack:
        x, y = stack.pop()

        if visited[x, y]:
            continue

        visited[x, y] = True
        pixel_value = int(image[x, y])

        if abs(pixel_value - seed_value) <= threshold:
            segmented[x, y] = 255

            # Check 4 neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                    stack.append((nx, ny))

    return segmented

def mouse_callback(event, x, y, flags, param):
    global clicked, seed_point
    if event == cv2.EVENT_LBUTTONDOWN:
        seed_point = (y, x)  # Note: (row, col)
        clicked = True

cv2.namedWindow("Click to Select Seed Point")
cv2.setMouseCallback("Click to Select Seed Point", mouse_callback)

while True:
    cv2.imshow("Click to Select Seed Point", display_img)
    key = cv2.waitKey(1)

    if clicked:
        result = region_growing(img, seed_point, threshold)
        cv2.imshow("Region Grown", result)
        clicked = False

    if key == 27:  # ESC key to exit
        break

cv2.destroyAllWindows()
