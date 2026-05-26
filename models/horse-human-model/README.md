# Horses vs. Humans Classification using CNN 🐴🧍

A professional Convolutional Neural Network (CNN) project that classifies images into two categories: **Horses** or **Humans**. This project demonstrates the power of CNNs in binary image classification and showcases best practices for model training, data augmentation, and evaluation.

## 🚀 Project Overview
This project uses a deep learning model built with **TensorFlow** and **Keras** to identify whether an image contains a horse or a human. It features a multi-layer CNN architecture designed to extract complex features from 300x300 RGB images.

## 🧠 CNN Architecture
The model consists of the following layers:
- **Convolutional Layers**: 3 layers with increasing filters (16, 32, 64) using `3x3` kernels and `ReLU` activation to extract spatial features.
- **Pooling Layers**: `MaxPooling2D` with `2x2` pools to reduce dimensionality and computation.
- **Flatten Layer**: Converts 2D feature maps into a 1D vector.
- **Dense Layers**: A fully connected layer with 128 neurons followed by a single output neuron with `Sigmoid` activation for binary classification.

## 📊 Dataset
The dataset used is the **Horses or Humans** dataset by Laurence Moroney, which contains computer-generated images.
- **Training Set**: 1,027 images (500 horses, 527 humans).
- **Validation Set**: 256 images (128 horses, 128 humans).
- **Source**: [Horses or Humans Dataset](https://www.tensorflow.org/datasets/catalog/horses_or_humans)

## 📈 Training Results
- **Epochs**: 15
- **Optimizer**: RMSprop
- **Loss Function**: Binary Crossentropy
- **Accuracy**: Achieved high training and validation accuracy (~90%+).

## 💡 Concepts Learned
- **Image Preprocessing**: Using `ImageDataGenerator` for rescaling and batch loading.
- **Feature Extraction**: How filters in CNNs learn edges, textures, and object parts.
- **Binary Classification**: Implementing the sigmoid activation function for 0/1 output.
- **Model Checkpointing**: Saving only the best-performing model during training.

## 🛠️ How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/kabeermemon03/tinyml-practice.git
cd tinyml-practice
```

### 2. Set Up Environment
Ensure you have **Python 3.12** installed.
```bash
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/macOS
pip install -r requirements.txt
```

### 3. Run Inference
You can test the model using the provided script:
```bash
python models/horse-human-model/test_horse_model.py
```
*(Note: You will need to provide a test image path in the script or place an image in the expected dataset directory.)*

## 📁 Project Structure
- `basics/horses-or-humans/`: Training script and logic.
- `models/horse-human-model/`: Contains the trained `.keras` model and inference script.
- `datasets/`: (Ignored in Git) Storage for training/validation images.
