import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import pathlib
from tqdm import tqdm


def format_image(image, label):
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image, label


def load_dataset():
    (raw_train, raw_validation, raw_test), metadata = tfds.load(
        'horses_or_humans',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )
    return (raw_train, raw_validation, raw_test), metadata


def create_model(num_classes, image_size=(224, 224)):
    base_model = keras.applications.MobileNetV2(
        input_shape=image_size + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = keras.Input(shape=image_size + (3,))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)


(raw_train, raw_validation, raw_test), metadata = load_dataset()

num_examples = metadata.splits['train'].num_examples
num_classes = metadata.features['label'].num_classes
print(f"Total examples: {num_examples}")
print(f"Number of classes: {num_classes}")

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

train_batches = raw_train.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = raw_validation.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = raw_test.map(format_image).batch(1)

print(f"Using MobileNetV2 with input size {IMAGE_SIZE}")

model = create_model(num_classes, IMAGE_SIZE)

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

EPOCHS = 5
model.fit(train_batches, epochs=EPOCHS, validation_data=validation_batches)

SAVED_MODEL_PATH = "exp_saved_model"
tf.saved_model.save(model, SAVED_MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

TFLITE_MODELS_DIR = pathlib.Path("models/tflite")
TFLITE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

tflite_model_file = TFLITE_MODELS_DIR / 'horses_humans_model.tflite'
tflite_model_file.write_bytes(tflite_model)

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

predictions = []
test_labels = []
test_imgs = []

for img, label in tqdm(test_batches.take(100)):
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(output_index))
    test_labels.append(label.numpy()[0])
    test_imgs.append(img)

correct = sum(1 for i in range(100) if np.argmax(predictions[i]) == test_labels[i])
print(f"Out of 100 predictions: {correct} correct")

CLASS_NAMES = ['horse', 'human']


def plot_image(index, predictions_array, true_labels, images):
    prediction_array = predictions_array[index]
    true_label = true_labels[index]
    image = images[index]
    
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    image = np.squeeze(image)
    plt.imshow(image, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(prediction_array)
    color = 'green' if predicted_label == true_label else 'red'
    confidence = 100 * np.max(prediction_array)
    
    plt.xlabel(
        f"{CLASS_NAMES[predicted_label]} {confidence:2.0f}% ({CLASS_NAMES[true_label]})",
        color=color
    )


max_visualizations = min(15, len(predictions))
for index in range(max_visualizations):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(index, predictions, test_labels, test_imgs)
    plot_file = TFLITE_MODELS_DIR / f'prediction_{index}.png'
    plt.savefig(plot_file)
    plt.close()

print("\nScript completed successfully!")
print(f"TFLite model saved to: {tflite_model_file}")