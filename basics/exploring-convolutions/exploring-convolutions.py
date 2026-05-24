import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import ascent

image = ascent().astype('int32')

plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()

transformed_image = np.copy(image)

filter = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]

weight = 1

for i in range(1, image.shape[0] - 1):
    for j in range(1, image.shape[1] - 1):

        convolution = 0.0

        convolution += image[i - 1, j - 1] * filter[0][0]
        convolution += image[i, j - 1] * filter[1][0]
        convolution += image[i + 1, j - 1] * filter[2][0]

        convolution += image[i - 1, j] * filter[0][1]
        convolution += image[i, j] * filter[1][1]
        convolution += image[i + 1, j] * filter[2][1]

        convolution += image[i - 1, j + 1] * filter[0][2]
        convolution += image[i, j + 1] * filter[1][2]
        convolution += image[i + 1, j + 1] * filter[2][2]

        convolution = convolution * weight

        if convolution < 0:
            convolution = 0

        elif convolution > 255:
            convolution = 255

        transformed_image[i, j] = convolution

plt.imshow(transformed_image, cmap='gray')
plt.title("Vertical Edge Detection")
plt.axis('off')
plt.savefig("images/vertical_edges.png")
print("Image saved successfully")
