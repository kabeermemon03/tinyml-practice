# TinyML Practice 🚀

A professional repository dedicated to learning and implementing **Tiny Machine Learning (TinyML)** on resource-constrained devices. This project explores the intersection of machine learning and embedded systems, focusing on creating and deploying highly efficient models to microcontrollers.

## 🛠️ Project Goals
- Understand the fundamentals of Machine Learning and Deep Learning.
- Learn model optimization techniques (Quantization, Pruning).
- Implement TensorFlow Lite Micro for embedded deployment.
- Develop end-to-end Edge AI applications for real-world scenarios.

---

## 📂 Repository Structure

| Directory | Description |
| :--- | :--- |
| **[basics/](basics)** | Fundamental ML concepts, simple models, and manual implementations. |
| **[tensorflow-lite/](tensorflow-lite)** | TFLite Micro implementations and model conversion workflows. |
| **[GridGaurd_TinyML/](GridGaurd_TinyML)** | End-to-end temperature anomaly detection project (data → training → TFLite → INT8 deployment). |
| **[datasets/](datasets)** | Curated datasets optimized for small-scale model training. |
| **[notebooks/](notebooks)** | Notes and experimentation for model training. |
| **[models/](models)** | Trained models, TFLite files, and evaluation artifacts. |
| **[microcontroller-deployment/](microcontroller-deployment)** | MCU-specific code (ESP32, STM32, Arduino) for model execution. |
| **[docs/](docs)** | Research notes, tutorials, and deep-dives into TinyML topics. |

---

## ⚙️ Environment Setup

This project requires **Python 3.12** due to current TensorFlow compatibility requirements on Windows.

### 1. Clone the repository
```bash
git clone https://github.com/kabeermemon03/tinyml-practice.git
cd tinyml-practice
```

### 2. Create a Virtual Environment
```bash
# Using Python 3.12
py -3.12 -m venv venv
```

### 3. Activate the Environment
- **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
- **Linux/macOS:** `source venv/bin/activate`

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Current Progress

### GridGuard AI - Temperature Anomaly Detection
An end-to-end TinyML project that detects abnormal temperature patterns (overheating grid / industrial equipment) and prepares a model for microcontroller deployment:

1. **`scripts/generate_dataset.py`** - Generates a synthetic labelled temperature dataset (normal vs. anomaly windows).
2. **`scripts/prepare_data.py`** - Splits the data (70/15/15), standardizes it, and caches it as NumPy arrays.
3. **`scripts/train_model.py`** - Trains a small MLP classifier (20 inputs → 32 → 16 → 1) with early stopping.
4. **`scripts/convert_tflite.py`** - Exports a **float32** TensorFlow Lite model.
5. **`scripts/quantize_int8.py`** - Applies **full INT8 post-training quantization** for MCU deployment.
6. **`scripts/validate_tflite.py` / `scripts/validate_int8.py`** - Verify the converted models against the Keras baseline.

All settings (paths, window size, thresholds) live in a single **`config.py`**, so every script works regardless of where the repository is checked out.

### Basics
- **CNN Edge Detection Visualization**: Understanding the math behind CNNs by implementing image filters from scratch.
- **Fashion MNIST CNN**: Image classification using convolutional neural networks.
- **Horses vs Humans**: Data augmentation, TFDS, and CNN training.
- **Post-Training Quantization**: Shrinking models for on-device inference.

---

## 🤝 Contributing
Contributions are welcome! If you have a TinyML project or optimization technique to share, feel free to open a PR.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.