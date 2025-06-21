"""
EC7212 – Computer Vision and Image Processing
Take Home Assignment 1 - Task 2
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def apply_mean_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load image in grayscale
    image_path = os.path.join(project_root, "Images", "task_02", "02.jpg")
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Apply averaging filters with different kernel sizes
    filtered_3x3 = apply_mean_filter(original_image, 3)
    filtered_10x10 = apply_mean_filter(original_image, 10)
    filtered_20x20 = apply_mean_filter(original_image, 20)
    
    # Prepare data for visualization
    image_titles = ['Original Image', '3×3 Avg Filter', '10×10 Avg Filter', '20×20 Avg Filter']
    processed_images = [original_image, filtered_3x3, filtered_10x10, filtered_20x20]
    
    # Display the results
    plt.figure(figsize=(16, 4))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(processed_images[i], cmap='gray')
        plt.title(image_titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Create output directory and save results
    results_directory = os.path.join(project_root, "Results", "task_02")
    os.makedirs(results_directory, exist_ok=True)
    
    # Save smoothed images
    cv2.imwrite(os.path.join(results_directory, "3x3_kernel.png"), filtered_3x3)
    cv2.imwrite(os.path.join(results_directory, "10x10_kernel.png"), filtered_10x10)
    cv2.imwrite(os.path.join(results_directory, "20x20_kernel.png"), filtered_20x20)
    
    print("Spatial averaging process complete!")
    print(f"Original dimensions: {original_image.shape[1]}×{original_image.shape[0]} pixels")
    print("\nKernel sizes applied:")
    print("  → 3×3 avg filter (light)")
    print("  → 10×10 avg filter (moderate)")
    print("  → 20×20 avg filter (heavy)")
    print(f"Output files saved to: {results_directory}")


if __name__ == "__main__":
    main()