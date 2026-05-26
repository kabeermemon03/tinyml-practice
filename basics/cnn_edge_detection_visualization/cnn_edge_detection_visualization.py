import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Load Fashion MNIST data for visualization
fashion_mnist = tf.keras.datasets.fashion_mnist
(_, _), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize and reshape
test_images = test_images / 255.0
val_images = test_images.reshape(10000, 28, 28, 1)

# Define the CNN model (matching the architecture in fashion-mnist-cnn.py)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Function to display original image
def show_image(img_idx):
    plt.figure()
    plt.imshow(val_images[img_idx].reshape(28,28), cmap='gray')
    plt.title(f"Original Image (Index {img_idx})")
    plt.grid(False)
    plt.show()

# Create 3x2 visualization grid
f, axarr = plt.subplots(3, 2, figsize=(10, 15))

# Select three images for visualization (e.g., shoe images)
FIRST_IMAGE = 0
SECOND_IMAGE = 23
THIRD_IMAGE = 28

# Select filter number (0-63)
CONVOLUTION_NUMBER = 1

# Collect outputs from CNN layers
layer_outputs = [layer.output for layer in model.layers]

# Create activation visualization model
activation_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=layer_outputs
)

print("Visualizing convolutions...")

for x in range(0, 2):
    # Predict for First Image
    f1 = activation_model.predict(val_images[FIRST_IMAGE].reshape(1,28,28,1))[x]
    axarr[0,x].imshow(f1[0,:,:,CONVOLUTION_NUMBER], cmap='inferno')
    axarr[0,x].set_title(f"Image {FIRST_IMAGE} - Layer {x+1}")
    axarr[0,x].grid(False)

    # Predict for Second Image
    f2 = activation_model.predict(val_images[SECOND_IMAGE].reshape(1,28,28,1))[x]
    axarr[1,x].imshow(f2[0,:,:,CONVOLUTION_NUMBER], cmap='inferno')
    axarr[1,x].set_title(f"Image {SECOND_IMAGE} - Layer {x+1}")
    axarr[1,x].grid(False)

    # Predict for Third Image
    f3 = activation_model.predict(val_images[THIRD_IMAGE].reshape(1,28,28,1))[x]
    axarr[2,x].imshow(f3[0,:,:,CONVOLUTION_NUMBER], cmap='inferno')
    axarr[2,x].set_title(f"Image {THIRD_IMAGE} - Layer {x+1}")
    axarr[2,x].grid(False)

# Save the visualization
output_dir = os.path.join(os.path.dirname(__file__), "images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.tight_layout()
output_path = os.path.join(output_dir, "cnn_activations.png")
plt.savefig(output_path)
print(f"Activation visualization saved to {output_path}")

# show_image(FIRST_IMAGE)
# show_image(SECOND_IMAGE)
# show_image(THIRD_IMAGE)

plt.close()
