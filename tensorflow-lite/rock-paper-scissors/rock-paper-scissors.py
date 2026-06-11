import subprocess
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import pathlib
from tqdm import tqdm

def format_image(image, label):
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image, label

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'rock_paper_scissors',
    split=['train[:80%]', 'train[80%:]', 'test'],
    with_info=True,
    as_supervised=True,
) 

num_examples = metadata.splits['train'].num_examples
num_classes = metadata.features['label'].num_classes 

BATCH_SIZE = 32

train_batches = raw_train.shuffle(
    num_examples // 4
).map(
    format_image
).batch(
    BATCH_SIZE
).prefetch(1)


validation_batches = raw_validation.map(
    format_image
).batch(
    BATCH_SIZE
).prefetch(1)

test_batches = raw_test.map(
    format_image
).batch(1)

for image_batch, label_batch in train_batches.take(1):
    pass

module_selection = ("mobilenet_v2", 224, 1280)
handle_base, pixels, FV_SIZE = module_selection
print(handle_base)

MODULE_HANDLE = "https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {} and output dimension {}".format(MODULE_HANDLE, IMAGE_SIZE, FV_SIZE))

class HubLayer(tf.keras.layers.Layer):
    def __init__(self, module_url, trainable=False):
        super(HubLayer, self).__init__()
        self.module = hub.load(module_url)
        self.trainable = trainable
    
    def call(self, inputs):
        return self.module(inputs)

# Create custom hub layer
hub_layer = HubLayer(MODULE_HANDLE, trainable=False)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=IMAGE_SIZE + (3,)),
    hub_layer,
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
)

EPOCHS = 5

hist = model.fit(train_batches,
                 epochs=EPOCHS,
                 validation_data=validation_batches)

ROCK_PAPER_SCISSORS_SAVED_MODEL = "exp_saved_model"
tf.saved_model.save(model, ROCK_PAPER_SCISSORS_SAVED_MODEL)

converter = tf.lite.TFLiteConverter.from_saved_model(
    ROCK_PAPER_SCISSORS_SAVED_MODEL
)

tflite_model = converter.convert()
tflite_models_dir = pathlib.Path("models")
tflite_models_dir.mkdir(exist_ok=True)

tflite_model_file = tflite_models_dir / "rock_paper_scissors_model.tflite"
tflite_model_file.write_bytes(tflite_model)

print("Saved:", tflite_model_file)

interpreter = tf.lite.Interpreter(
    model_path="models/rock_paper_scissors_model.tflite"
)

# Save the Keras model as well
model.save('rock_paper_scissors_model.keras')

predictions = []
test_labels = []
test_imgs = []

# Use the Keras model directly for inference (TFLite doesn't support custom layers)
print("\nRunning inference on test set...")
for img, label in tqdm(test_batches.take(100)):
    pred = model(img, training=False)
    predictions.append(pred.numpy())    
    test_labels.append(label.numpy()[0])
    test_imgs.append(img.numpy())

score = 0 
for item in range(0, 100):
    prediction = np.argmax(predictions[item])
    label = test_labels[item]
    if prediction==label:
        score=score+1

print("\nOut of 100 predictions I got " + str(score) + " correct")

class_names = ['rock', 'paper', 'scissors']

def plot_image(i,
                predictions_array,
                true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    img = np.squeeze(img)

    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]), color=color)

max_index = 73 
for index in range(0, max_index):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(index, predictions, test_labels, test_imgs)
    plt.show()