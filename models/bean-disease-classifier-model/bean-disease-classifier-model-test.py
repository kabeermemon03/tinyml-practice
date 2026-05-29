import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

model = tf.keras.models.load_model(
    "models/bean-disease-classifier-model/bean-disease-classifier-model.keras"
)

class_names = [
    "Angular Leaf Spot",
    "Bean Rust",
    "Healthy"
]

img_path = "datasets/bean-disease-classifier/test/angular_leaf_spot/angular_leaf_spot_test.29.jpg"

img = image.load_img(
    img_path,
    target_size=(224,224)
)

img_array = image.img_to_array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]

confidence = np.max(prediction) * 100

plt.figure(figsize=(6,6))

plt.imshow(img)

plt.title(
    f"{predicted_class}\nConfidence: {confidence:.2f}%",
    fontsize=16
)

plt.axis("off")

plt.show()

print("Prediction:", predicted_class)

print("Raw Prediction:", prediction)

