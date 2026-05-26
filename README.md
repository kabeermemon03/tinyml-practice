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
| **[basics/](file:///b:/tinyml-practice/basics)** | Fundamental ML concepts, simple models, and manual implementations. |
| **[tensorflow-lite/](file:///b:/tinyml-practice/tensorflow-lite)** | TFLite Micro implementations and model conversion workflows. |
| **[edge-ai-projects/](file:///b:/tinyml-practice/edge-ai-projects)** | Production-ready Edge AI applications and prototypes. |
| **[datasets/](file:///b:/tinyml-practice/datasets)** | Curated datasets optimized for small-scale model training. |
| **[notebooks/](file:///b:/tinyml-practice/notebooks)** | Interactive Jupyter notebooks for experimentation and training. |
| **[microcontroller-deployment/](file:///b:/tinyml-practice/microcontroller-deployment)** | MCU-specific code (ESP32, STM32, Arduino) for model execution. |
| **[docs/](file:///b:/tinyml-practice/docs)** | Research notes, tutorials, and deep-dives into TinyML topics. |

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

### Basics
- **MNIST Classification**: A simple dense neural network for digit recognition.
- **Manual Convolutions**: Understanding the math behind CNNs by implementing image filters from scratch.

---

## 🤝 Contributing
Contributions are welcome! If you have a TinyML project or optimization technique to share, feel free to open a PR.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
