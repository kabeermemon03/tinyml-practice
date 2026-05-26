import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

def load_and_predict(model_path, image_path):
    """
    Loads a trained Keras model and predicts if the image is a Horse or a Human.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the image
    print(f"Processing image: {image_path}...")
    img = image.load_img(image_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    
    # Interpret result (Binary classification: > 0.5 is Human, < 0.5 is Horse)
    result = "Human" if prediction[0] > 0.5 else "Horse"
    confidence = prediction[0][0] if prediction[0] > 0.5 else 1 - prediction[0][0]
    
    print(f"\nPrediction: {result}")
    print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    # Define paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "best_horse_human_model.keras")
    
    # Example test image (update this path to a local image for testing)
    TEST_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "datasets", "horses-or-humans", "cartoon-horse-test.png")

    load_and_predict(MODEL_PATH, TEST_IMAGE_PATH)
