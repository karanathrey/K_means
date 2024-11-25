import numpy as np
import os



# Function to perform image compression using Agglomerative Clustering
def agglomerative_compression(image, k, c):
    # Process the image into flattened windows
    data = process_image(image, c)
    
    # Apply Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=k, linkage='complete')
    labels = clustering.fit_predict(data)
    
    # Manually compute the cluster centers (mean of points in each cluster)
    centers = np.zeros((k, data.shape[1]))
    for i in range(k):
        centers[i] = data[labels == i].mean(axis=0)
    
    # Map each window to the nearest cluster center
    compressed_data = centers[labels]
    
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
    for i in range(k):
        # Extract all data points that belong to the current cluster
        cluster_points = data[labels == i]
        # Compute squared Euclidean distance from each point in the cluster to the centroid
        if cluster_points.size > 0:
            distances = np.linalg.norm(cluster_points - centers[i], axis=1) ** 2
            total_reconstruction_error += np.sum(distances)


    return compressed_image, total_reconstruction_error


k_list = [4, 8, 16]
c_list = [20, 40, 60]

min_error = float('inf')
min_compression = float('inf')
best_k = 0
best_c = 0

for k in k_list:
    for c in c_list:

        compressed_jpg, error = agglomerative_compression(image_array, k, c)
        
        # Ensure the image data is in the correct format
        compressed_jpg_uint8 = compressed_jpg.astype(np.uint8)

        filename = f'Q3_compressed_image_k{k}_c{c}.jpg'
        
        # Save the image using skimage
        io.imsave(filename, compressed_jpg_uint8)
        
        # Display the image using matplotlib
        plt.figure(figsize=(10, 8))
        plt.imshow(compressed_jpg_uint8)
        plt.title(f"Compressed Image with k={k}, c={c}")
        plt.axis('off')
        plt.show()

        # Calculate compression rate
        original_size = os.path.getsize('Assignment 2_image.jpg')
        compressed_size = os.path.getsize(filename)
        compression_rate = original_size / compressed_size
        
        print("Reconstruction error:", error)
        print("Compression Rate:", compression_rate)

        if error < min_error and compression_rate < min_compression:
            min_error = error
            min_compression = compression_rate
            best_k = k
            best_c = c
