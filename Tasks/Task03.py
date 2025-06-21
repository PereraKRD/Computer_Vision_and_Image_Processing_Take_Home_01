"""
EC7212 – Computer Vision and Image Processing
Take Home Assignment 1 - Task 3
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def rotate_image(image, rotation_angle):
    image_height, image_width = image.shape
    center_point = (image_width // 2, image_height // 2)
    
    # Compute rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center_point, rotation_angle, scale=1.0)
    
    # Compute the new bounding dimensions of the image
    cos_angle = np.abs(rotation_matrix[0, 0])
    sin_angle = np.abs(rotation_matrix[0, 1])
    new_width = int((image_height * sin_angle) + (image_width * cos_angle))
    new_height = int((image_height * cos_angle) + (image_width * sin_angle))
    
    # Adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (new_width / 2) - center_point[0]
    rotation_matrix[1, 2] += (new_height / 2) - center_point[1]
    
    # Rotate the image with white background
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), borderValue=255)
    return rotated_image


def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_directory)
    
    # Load image (grayscale for simplicity)
    input_image_path = os.path.join(project_root, "Images", "task_03", "03.jpg")
    original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_image is None:
        raise FileNotFoundError(f"Image not found at {input_image_path}")
    
    # Rotate images by specified angles
    image_rotated_45 = rotate_image(original_image, -45)
    image_rotated_90 = rotate_image(original_image, -90)
    
    # Prepare data for visualization
    display_titles = ['Original Image', 'Rotated 45°', 'Rotated 90°']
    display_images = [original_image, image_rotated_45, image_rotated_90]
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(display_images[i], cmap='gray')
        plt.title(display_titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Create output directory and save results
    output_directory = os.path.join(project_root, "Results", "task_03")
    os.makedirs(output_directory, exist_ok=True)
    
   # Save transformed images
    cv2.imwrite(os.path.join(output_directory, "transform_45deg.png"), image_rotated_45)
    cv2.imwrite(os.path.join(output_directory, "transform_90deg.png"), image_rotated_90)
    
    print("Rotation processing complete!")
    print(f"Original dimensions: {original_image.shape[1]}×{original_image.shape[0]} pixels")
    print(f"45° rotation result: {image_rotated_45.shape[1]}×{image_rotated_45.shape[0]} pixels")
    print(f"90° rotation result: {image_rotated_90.shape[1]}×{image_rotated_90.shape[0]} pixels")
    print("\nTransformations applied:")
    print("  → 45° counter-clockwise rotation")
    print("  → 90° counter-clockwise rotation")
    print(f"Output files saved to: {output_directory}")


if __name__ == "__main__":
    main()