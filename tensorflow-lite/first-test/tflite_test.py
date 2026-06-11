import tensorflow as tf 
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pathlib

l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0], dtype=np.float32)
model.fit(xs, ys, epochs=500)

print(model.predict(np.array([10.0])))
print("Here is what I learned: {}".format(l0.get_weights()))

export_dir = './models/tflite-models/1'
tf.saved_model.save(model, export_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

tflite_model_file = pathlib.Path('model.tflite')
tflite_model_file.write_bytes(tflite_model)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)

to_predict = np.array([[10.0]], dtype=np.float32)
print(to_predict)
interpreter.set_tensor(input_details[0]['index'], to_predict)
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])
print(tflite_results)

