import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import ascent
import os

def apply_convolution(image, kernel, weight=1):
    """
    Applies a 3x3 convolution kernel to an image.
    
    Args:
        image: 2D numpy array representing the grayscale image.
        kernel: 3x3 list or numpy array representing the convolution filter.
        weight: A multiplier for the final convolution result.
        
    Returns:
        A new image with the convolution applied.
    """
    size_x = image.shape[0]
    size_y = image.shape[1]
    transformed_image = np.copy(image)

    for x in range(1, size_x - 1):
        for y in range(1, size_y - 1):
            convolution = 0.0
            # Manually apply the 3x3 kernel
            convolution += (image[x - 1, y - 1] * kernel[0][0])
            convolution += (image[x, y - 1] * kernel[1][0])
            convolution += (image[x + 1, y - 1] * kernel[2][0])
            convolution += (image[x - 1, y] * kernel[0][1])
            convolution += (image[x, y] * kernel[1][1])
            convolution += (image[x + 1, y] * kernel[2][1])
            convolution += (image[x - 1, y + 1] * kernel[0][2])
            convolution += (image[x, y + 1] * kernel[1][2])
            convolution += (image[x + 1, y + 1] * kernel[2][2])
            
            convolution = convolution * weight
            
            # Clamp values to [0, 255]
            if convolution < 0:
                convolution = 0
            if convolution > 255:
                convolution = 255
            
            transformed_image[x, y] = convolution
            
    return transformed_image

def main():
    # Load a sample image
    image = ascent().astype('int32')
    
    # Define a vertical edge detection filter (Sobel)
    vertical_filter = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    
    print("Applying vertical edge detection filter...")
    transformed_image = apply_convolution(image, vertical_filter)
    
    # Create images directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "images")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image, cmap='gray')
    plt.title("Vertical Edge Detection")
    plt.axis('off')
    
    output_path = os.path.join(output_dir, "comparison.png")
    plt.savefig(output_path)
    print(f"Comparison saved successfully to: {output_path}")
    # plt.show() # Optional: display the plot

if __name__ == "__main__":
    main()
