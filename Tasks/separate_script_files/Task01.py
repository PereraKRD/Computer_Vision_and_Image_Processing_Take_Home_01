"""
EC7212 – Computer Vision and Image Processing
Take Home Assignment 1 - Task 1
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def reduce_intensity_levels(image, levels):
    assert (levels & (levels - 1) == 0) and levels <= 256, "Levels must be power of 2 and <= 256"
    factor = 256 // levels
    reduced_img = (image // factor) * factor
    return reduced_img


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load the image
    img_path = os.path.join(project_root, "Images", "task_01", "01.jpg")
    original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if original_img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    
    # Intensity level variable initialization
    desired_levels = 4
    
    # Process image
    reduced_img = reduce_intensity_levels(original_img, desired_levels)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(project_root, "Results", "task_01")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the processed image
    output_path = os.path.join(output_dir, f"quantized_{desired_levels}_levels.png")
    cv2.imwrite(output_path, reduced_img)
    
    # Display the results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Reduced to {desired_levels} Levels")
    plt.imshow(reduced_img, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Intensity reduction process complete!")
    print(f"Original dimensions: {original_img.shape[1]}×{original_img.shape[0]} pixels")
    print(f"Intensity levels reduced: 256 → {desired_levels} levels")
    print(f"Output file saved to: {output_path}")

if __name__ == "__main__":
    main()