import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)
from tensorflow import keras
import pathlib


def quantizeAndReconstruct(weights):
    max_weight = np.max(weights)
    min_weight = np.min(weights)
    range = max_weight - min_weight

    max_int8 = 2**8

    scale = range / max_int8

    midpoint = np.mean([max_weight, min_weight])
    centered_weights = weights - midpoint
    quantized_weights = np.rint(centered_weights / scale )

    reconstructed_weights = scale * quantized_weights + midpoint
    return reconstructed_weights

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(28,28)),
    keras.layers.Reshape(target_shape=(28,28,1)),
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10),
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    train_images,
    train_labels,
    epochs=1,
    validation_data=(test_images, test_labels)
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_models_dir = pathlib.Path("models/tflite-models")
tflite_model_file = tflite_models_dir/"mnist_model.tflite"
tflite_model_file.write_bytes(tflite_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
tflite_model_quant_file = tflite_models_dir / "mnist_model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)


