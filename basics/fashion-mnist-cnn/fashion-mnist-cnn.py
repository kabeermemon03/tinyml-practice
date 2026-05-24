import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(training_images.shape)

plt.imshow(training_images[0], cmap='gray')
plt.title("Sample Fashion MNIST Image")
plt.axis('off')
plt.show()

print(training_labels[0])

training_images = training_images / 255.0
test_images = test_images / 255.0

training_images = training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)


model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(
        64,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)
    ),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(
        64,(3,3),activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    training_images,
    training_labels,
    epochs=5
)

test_loss, test_accuracy = model.evaluate(
    test_images,
    test_labels
)

print("Test Accuracy:", test_accuracy)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

predictions = model.predict(test_images)

plt.figure(figsize=(6,6))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(test_images[i].reshape(28,28), cmap='gray')

    predicted_label = class_names[np.argmax(predictions[i])]
    true_label = class_names[test_labels[i]]

    plt.title(f"P:{predicted_label}\nT:{true_label}")

    plt.axis('off')

plt.tight_layout()
plt.show()
model.summary()

model.save("/models/fashion_mnist_cnn.h5")