import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io
import numpy as np


def process_image(image, c):
    # Get the dimensions of the image
    height, width, _ = image.shape
    
    # Initialize the list to hold the flattened windows
    flattened_windows = []
    
    # Loop over the image with step size equal to c
    for i in range(0, height, c):
        for j in range(0, width, c):
            # Extract the window
            if (i + c <= height) and (j + c <= width):
                window = image[i:i + c, j:j + c]
                # Flatten and store the window
                flattened_windows.append(window.flatten('F'))
    
    return np.array(flattened_windows)

def k_means_compression(image, k, c):
    # Process the image into flattened windows
    data = process_image(image, c)
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # Map each window to the nearest cluster center
    compressed_data = kmeans.cluster_centers_[kmeans.labels_]
    
    # Reconstruct the image from the compressed data
    idx = 0
    height, width, channels = image.shape
    compressed_image = np.zeros_like(image)
    for i in range(0, height, c):
        for j in range(0, width, c):
            if (i + c <= height) and (j + c <= width):
                window_shape = (c, c, channels)
                compressed_image[i:i+c, j:j+c] = \
                    compressed_data[idx].reshape(window_shape, order='F')
                idx += 1

    total_reconstruction_error = 0
    # Iterate over all data points and their assigned cluster labels
    for i, center in enumerate(centers):
        # Extract all data points that belong to the current cluster
        cluster_points = data[labels == i]
        # Compute squared Euclidean distance from each point in the cluster to the centroid
        if cluster_points.size > 0:
            distances = np.linalg.norm(cluster_points - center, axis=1) ** 2
            total_reconstruction_error += np.sum(distances)

    return compressed_image, total_reconstruction_error


# Load the image
image_path = 'Assignment 2_image.jpg'
image_array = io.imread(image_path)

# Display image dimensions
print(image_array.shape)

import matplotlib.pyplot as plt

k_list = [4, 8, 16]
c_list = [20, 40, 60]

min_error = float('inf')
best_k = 0
best_c = 0

for k in k_list:
    for c in c_list:

        compressed_jpg, error = k_means_compression(image_array, k, c)

        # Ensure the image data is in the correct format
        compressed_jpg_uint8 = compressed_jpg.astype(np.uint8)

        # Save the image using skimage
        io.imsave(f'Q1_compressed_image_k{k}_c{c}.jpg', compressed_jpg_uint8)

        # Display the image using matplotlib
        plt.figure(figsize=(10, 8))
        plt.imshow(compressed_jpg_uint8)
        plt.title(f"Compressed Image with k={k}, c={c}")
        plt.axis('off')
        plt.show()

        print("Reconstruction error:", error)

        if error < min_error:
            min_error = error
            best_k = k
            best_c = c
