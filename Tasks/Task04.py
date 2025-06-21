"""
EC7212 – Computer Vision and Image Processing
Take Home Assignment 1 - Task 4
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def block_average(input_image, block_size):
    image_height, image_width = input_image.shape
    averaged_image = input_image.copy()
    
    # Process non-overlapping blocks
    for row in range(0, image_height - image_height % block_size, block_size):
        for col in range(0, image_width - image_width % block_size, block_size):
            # Extract current block
            current_block = input_image[row:row + block_size, col:col + block_size]
            
            # Calculate average value of the block
            block_average_value = np.mean(current_block, dtype=np.float32)
            
            # Replace all pixels in the block with the average value
            averaged_image[row:row + block_size, col:col + block_size] = int(block_average_value)
    
    return averaged_image


def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_directory)
    
    # Load grayscale image
    input_image_path = os.path.join(project_root, "Images", "task_04", "04.jpg")
    original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_image is None:
        raise FileNotFoundError(f"Image not found at {input_image_path}")
    
    # Apply block averaging for different block sizes
    block_averaged_3x3 = block_average(original_image, 3)
    block_averaged_5x5 = block_average(original_image, 5)
    block_averaged_7x7 = block_average(original_image, 7)
    
    # Prepare data for visualization
    display_titles = ['Original Image', '3×3 Spatial Reduction', '5×5 Spatial Reduction', '7×7 Spatial Reduction']
    processed_images = [original_image, block_averaged_3x3, block_averaged_5x5, block_averaged_7x7]
    
    # Display the results
    plt.figure(figsize=(16, 4))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(processed_images[i], cmap='gray')
        plt.title(display_titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Create output directory and save results
    output_directory = os.path.join(project_root, "Results", "task_04")
    os.makedirs(output_directory, exist_ok=True)
    
    # Save processed images with block averaging
    cv2.imwrite(os.path.join(output_directory, "spatial_reduced_3x3.png"), block_averaged_3x3)
    cv2.imwrite(os.path.join(output_directory, "spatial_reduced_5x5.png"), block_averaged_5x5)
    cv2.imwrite(os.path.join(output_directory, "spatial_reduced_7x7.png"), block_averaged_7x7)
    
    print("Spatial resolution reduction process complete!")
    print(f"Input image dimensions: {original_image.shape[1]}x{original_image.shape[0]} pixels")
    print("\nBlock sizes processed:")
    print("  → 3x3 pixel blocks")
    print("  → 5x5 pixel blocks") 
    print("  → 7x7 pixel blocks")
    print(f"Output files saved to: {output_directory}")
    
    # Display resolution reduction statistics
    total_pixels = original_image.shape[0] * original_image.shape[1]
    print(f"\nResolution reduction analysis:")
    print(f"Total original pixels: {total_pixels:,}")
    print(f"3x3 averaging: ~{total_pixels//9:,} effective pixels (9x reduction)")
    print(f"5x5 averaging: ~{total_pixels//25:,} effective pixels (25x reduction)")
    print(f"7x7 averaging: ~{total_pixels//49:,} effective pixels (49x reduction)")


if __name__ == "__main__":
    main()