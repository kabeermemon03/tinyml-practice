import tensorflow as tf
import os

def load_data():
    """Load and normalize the MNIST dataset."""
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    
    return (training_images, training_labels), (test_images, test_labels)

def create_model():
    """Create a simple sequential model for digit classification."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28), name="input_layer"),
        tf.keras.layers.Dense(128, activation='relu', name="hidden_layer"),
        tf.keras.layers.Dense(10, activation='softmax', name="output_layer")
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("TensorFlow version:", tf.__version__)
    
    # Load data
    (training_images, training_labels), (test_images, test_labels) = load_data()
    
    # Create and train model
    model = create_model()
    model.summary()
    
    print("\nStarting training...")
    model.fit(training_images, training_labels, epochs=5)
    
    # Evaluate model
    print("\nEvaluating model on test data...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), "mnist_model.h5")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main()
