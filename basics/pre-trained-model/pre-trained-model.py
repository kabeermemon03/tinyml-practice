import tensorflow.compat.v1 as tf
import sys
# from IPython.display import HTML,Audio  # Colab-only
# sys.path.append("/content/tensorflow/tensorflow/examples/speech_commands/")  # Colab-only
# import input_data  # Colab-only
# import models  # Colab-only
import numpy as np 
import pickle
import os

WANTED_WORDS = "yes,no"
print("Spotting these words is: %s % WANTED_WORDS")

number_of_labels = WANTED_WORDS.count(',') + 1  # 1 + 1 = 2 or to show the numbe rof labels are two by putting comma + 1 
number_of_total_labels = number_of_labels + 2   # Google speech commands will add two more labels are [silence, unknown], 2 + 2 = 4
equal_percentage_of_training_samples = int(100.0/(number_of_total_labels))   # This will help balance training by 25 % each
SILENT_PERCENTAGE = equal_percentage_of_training_samples
UNKNOWN_PERCENTAGE = equal_percentage_of_training_samples

# Constants which are shared during training and inference
PREPROCESS = 'micro'
WINDOW_STRIDE = 20
MODEL_ARCHITECTURE = 'tiny_conv'

# Constants for training directories and filepaths
DATASET_DIR = "datasets/speech-commands"
LOGS_DIR = 'logs/'
TRAIN_DIR = 'train/' # for training checkpoints and other files.

# Constants for inference directories and filepaths
MODELS_DIR = 'models/speech-commands'
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_TF = os.path.join(MODELS_DIR, 'model.pb')
MODEL_TFLITE = os.path.join(MODELS_DIR, 'model.tflite')
FLOAT_MODEL_TFLITE = os.path.join(MODELS_DIR, 'float_model.tflite')
MODEL_TFLITE_MICRO = os.path.join(MODELS_DIR, 'model.cc')
SAVED_MODEL = os.path.join(MODELS_DIR, 'saved_model')

# Constants for Quantization
QUANT_INPUT_MIN = 0.0
QUANT_INPUT_MAX = 26.0
QUANT_INPUT_RANGE = QUANT_INPUT_MAX - QUANT_INPUT_MIN

# Constants for audio process during Quantization and Evaluation
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME_RANGE = 0.1
TIME_SHIFT_MS = 100.0

# URL for the dataset and train/val/test split
DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10

model_settings = models.prepare_model_settings(
  len(input_data.prepare_words_list(WANTED_WORDS.split(','))),
  SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
  WINDOW_STRIDE, FEATURE_BIN_COUNT, PREPROCESS)

audio_processor = input_data.AudioProcessor(
  DATA_URL, DATASET_URL,
  SILENT_PERCENTAGE, UNKNOWN_PERCENTAGE,
  WANTED_WORDS.split(','), VALIDATION_PERCENTAGE,
  TESTING_PERCENTAGE, model_settings, LOGS_DIR
)

with tf.Session() as sess:
  float_converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
  float_tflite_model = float_converter.convert()
  float_tflite_model_size = open(FLOAT_MODEL_TFLITE, 'wb').write(float_tflite_model)
  print("Float model is %d bytes" % float_tflite_model_size)

  converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.inference_input_type = tf.lite.constants.INT8
  converter.inference_output_type = tf.lite.constants.INT8

  def representative_dataset_gen():
    for i in range(100):
      data, _ = audio_processor.get_data(1, i, model_settings,
                                         BACKGROUND_FREQUENCY,
                                         BACKGROUND_VOLUME_RANGE,
                                         TIME_SHIFT_MS,
                                         'testing',
                                          sess)
      flattened_data = np.array(data.flatten(), dtype=np.float32).reshape(1, 1960)
      yield [flattened_data]

      converter.representative_dataset = representative_dataset_gen
      tflite_model = converter.convert()
      tflite_model_size = open(MODEL_TFLITE, 'wb').write(tflite_model)
      
      print("Quantized model is %d bytes" % tflite_model_size)

def run_tflite_inference_testSet(tflite_model_path, model_type="Float"):
  np.random.seed(0)
  with tf.Session() as sess:
    test_data, test_labels = audio_processor.get_data(
      -1, 0, model_settings, BACKGROUND_FREQUENCY, BACKGROUND_VOLUME_RANGE,
      TIME_SHIFT_MS, 'testing', sess)

  test_data = np.expand_dims(test_data, axis=1).astype(np.float32)

  interpreter = tf.lite.Interpreter(tflite_model_path)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]


  if model_type == 'Quantized':
    input_scale, input_zero_point = input_details["quantization"]
    test_data = test_data / input_scale + input_zero_point
    test_data= test_data.astype(input_details["dtype"])

  correct_predictions = 0 
  for i  in range(len(test_data)):
    interpreter.set_tensor(input_details["index"], test_data[i])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    top_prediction = output.argmax()
    correct_predictions += (top_prediction == test_labels[i])

  print('%s model accuracy is %f%% (Number of test samples=%d)' % (
     model_type, (correct_predictions * 100) / len(test_data), len(test_data)))

