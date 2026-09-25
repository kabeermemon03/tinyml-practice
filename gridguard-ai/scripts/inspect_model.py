import tensorflow as tf
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "anomaly_model.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n========== INPUT ==========")

for item in input_details:
    print("Name:", item["name"])
    print("Shape:", item["shape"])
    print("Type:", item["dtype"])
    print("Quantization:", item["quantization"])
    print("Quantization parameters:", item["quantization_parameters"])

print("\n========== OUTPUT ==========")

for item in output_details:
    print("Name:", item["name"])
    print("Shape:", item["shape"])
    print("Type:", item["dtype"])
    print("Quantization:", item["quantization"])
    print("Quantization parameters:", item["quantization_parameters"])