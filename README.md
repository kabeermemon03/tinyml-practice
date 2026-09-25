# TinyML Practice 🚀

<p align="center">
  <img src="https://img.shields.io/badge/TinyML-Edge%20AI-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/TensorFlow-Lite%20Micro-FF6F00?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/ESP32-Embedded-red?style=for-the-badge&logo=espressif">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python">
</p>

A professional repository documenting my journey into **Tiny Machine Learning (TinyML)** and **Edge AI**, focusing on designing, training, optimizing, and deploying machine learning models on resource-constrained embedded systems such as the ESP32 and Arduino platforms.

The goal of this repository is to bridge **Artificial Intelligence** and **Embedded Systems** by building practical TinyML applications suitable for real-world IoT and edge computing scenarios.

---

# 🎯 Current Focus

I am currently developing and optimizing TinyML models for our **IEEE HART HardwAIre Challenge** project.

The project focuses on building an **Edge AI-enabled Wireless Temperature Monitoring System** using ESP32-based sensor nodes, where TinyML is used for:

- Temperature trend prediction
- Thermal anomaly detection
- Sensor fault detection
- Adaptive sensing
- Intelligent edge-based decision making

The complete TinyML workflow—from dataset generation and preprocessing to TensorFlow Lite Micro deployment—is being developed inside this repository.

---

# 🛠 Repository Goals

- Learn Machine Learning fundamentals
- Learn Deep Learning fundamentals
- Build CNNs and Neural Networks
- Understand TensorFlow Lite
- Deploy TensorFlow Lite Micro models
- Learn model optimization
  - Quantization
  - Pruning
  - Model Compression
- Deploy models on ESP32
- Develop production-ready TinyML applications
- Explore real-world Edge AI systems

---

# 📂 Repository Structure

| Directory | Description |
|------------|-------------|
| **basics/** | Fundamental Machine Learning concepts and implementations |
| **tensorflow-lite/** | TensorFlow Lite and TensorFlow Lite Micro workflows |
| **gridguard-ai/** | GridGuard AI v1 - temperature anomaly detection (data → training → TFLite → INT8 deployment) |
| **gridguard-ai-v2/** | GridGuard AI v2 - ESP32-S3 temperature classification with INT8 quantization |
| **gridguard-fire-ai/** | GridGuard Fire AI - fire vs no-fire image classification CNN (ESP32-CAM) |
| **datasets/** | Datasets used for training TinyML models |
| **notebooks/** | Jupyter notebooks for experimentation |
| **models/** | Trained models, TFLite files and evaluation artifacts |
| **microcontroller-deployment/** | ESP32, Arduino and embedded deployment code |
| **docs/** | Research notes, tutorials and learning resources |

---

# 🚀 Featured Projects

### 📌 GridGuard AI - Temperature Anomaly Detection

An end-to-end TinyML project that detects abnormal temperature patterns and prepares a model for ESP32 / microcontroller deployment (`gridguard-ai/`):

1. **`gridguard-ai/scripts/generate_dataset.py`** - Generates a synthetic labelled temperature dataset (normal vs. anomaly windows).
2. **`gridguard-ai/scripts/prepare_data.py`** - Splits the data (70/15/15), standardizes it, and caches it as NumPy arrays.
3. **`gridguard-ai/scripts/train_model.py`** - Trains a small MLP classifier (20 → 32 → 16 → 1) with early stopping.
4. **`gridguard-ai/scripts/convert_tflite.py`** - Exports a **float32** TensorFlow Lite model.
5. **`gridguard-ai/scripts/quantize_int8.py`** - Applies **full INT8 post-training quantization** for MCU deployment.
6. **`gridguard-ai/scripts/validate_tflite.py` / `gridguard-ai/scripts/validate_int8.py`** - Verify the converted models against the Keras baseline.

All settings (paths, window size, thresholds) live in a single **`gridguard-ai/config.py`**, so every script works regardless of where the repository is checked out.

---

### 📌 GridGuard AI v2 - Temperature Classification (ESP32-S3)

Full 5-class temperature classification pipeline using 40-feature windows (baseline + current), INT8 quantization and TensorFlow Lite Micro inference on ESP32-S3. See `gridguard-ai-v2/README.md` and `gridguard-ai-v2/deployment/ESP32_DEPLOYMENT.md`.

---

### 📌 GridGuard Fire AI - Fire vs No-Fire (ESP32-CAM)

CNN image classification for fire detection with a leakage-safe stratified pipeline and ESP32-CAM deployment scripts. See `gridguard-fire-ai/README.md`.

---

### 📌 IEEE HART TinyML Temperature Monitoring *(In Progress)*

Developing TinyML models capable of running directly on ESP32 sensor nodes for intelligent temperature monitoring.

Current research includes:

- Temperature Classification
- Temperature Prediction
- Thermal Anomaly Detection
- Sensor Fault Detection
- Adaptive Sampling
- Edge Intelligence

---

### 📌 MNIST Classification

Simple dense neural network for handwritten digit recognition using TensorFlow.

---

### 📌 Manual Convolutions

Implementation of convolution operations from scratch to understand the mathematics behind CNNs.

---

# ⚙ Environment Setup

## Clone Repository

```bash
git clone https://github.com/kabeermemon03/tinyml-practice.git

cd tinyml-practice
```

## Create Virtual Environment

```bash
py -3.12 -m venv venv
```

## Activate

### Windows

```powershell
.\venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📈 Learning Roadmap

- [x] Python Fundamentals
- [x] NumPy
- [x] Pandas
- [x] TensorFlow Basics
- [x] CNN Fundamentals
- [x] TensorFlow Lite
- [ ] TensorFlow Lite Micro
- [ ] ESP32 Deployment
- [ ] Model Quantization
- [ ] Model Pruning
- [ ] Thermal Anomaly Detection
- [ ] Sensor Fault Detection
- [ ] Adaptive Sampling
- [ ] IEEE HART Edge AI Deployment

---

# 🧪 Technologies

- Python
- TensorFlow
- TensorFlow Lite
- TensorFlow Lite Micro
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook
- ESP32
- Arduino
- Edge AI
- TinyML

---

# 🤝 Contributing

Contributions are always welcome.

If you have TinyML examples, optimization techniques, deployment strategies, or Edge AI projects, feel free to open an issue or submit a pull request.

---

# 📚 References

- TensorFlow
- TensorFlow Lite
- TensorFlow Lite Micro
- Harvard TinyML Course
- TinyML Foundation
- Edge Impulse Documentation

---

# 📜 License

This project is licensed under the MIT License.

---

## ⭐ If you find this repository useful, consider giving it a star!

It motivates me to continue documenting my TinyML journey and sharing practical Edge AI implementations with the community.