# 🌌 Aetheris Neural Engine v2.1

> A minimalist, high-performance Deep Learning inference node.

---

## 🛰️ Project Overview

**Aetheris Neural Engine** is a standalone desktop application designed for real-time handwritten digit recognition. Unlike standard "Hello World" AI projects, Aetheris implements a Deep Convolutional Neural Network (CNN) with a custom-built, cyber-industrial terminal interface.

The project bridges the gap between high-level cloud training (Google Colab) and local low-latency execution (Linux/Debian), providing a seamless **"Draft-to-Deploy"** workflow.

---

## 🛠️ Technical Architecture: The "Why"

### 1. The Brain: Convolutional Neural Network (CNN)

Instead of a simple Multi-Layer Perceptron (MLP), I chose a CNN architecture.

- **Feature Extraction**: By using `Conv2d` layers, the model learns spatial hierarchies of features (edges → curves → shapes) rather than treating pixels as independent variables.
- **Batch Normalization**: I implemented `BatchNorm2d` after convolutional layers to stabilize the learning process and significantly accelerate convergence (reaching ~99% accuracy in just 2 epochs).
- **Dropout Regularization**: To prevent overfitting, `Dropout(0.5)` is used in the fully connected layers, ensuring the engine generalizes well to "messy" human handwriting.

### 2. The Engine: PyTorch & AdamW

- **PyTorch**: Selected for its dynamic computation graph, essential for rapid prototyping and clean model definitions.
- **AdamW Optimizer**: I opted for AdamW (Adam with Weight Decay) instead of standard SGD. It provides better regularization and handles the training dynamics of deep networks more efficiently.

### 3. The Interface: PyWebView & Canvas API

- **The Choice**: Instead of outdated libraries like Tkinter, Aetheris uses PyWebView. This allows for a modern, hardware-accelerated UI using HTML5/CSS3.
- **Zero-Latency Bridge**: The communication between the JavaScript Canvas (where the user draws) and the Python Backend (where the model lives) is handled via a base64 encoded bi-directional bridge.

---

## 🧬 System Structure

The repository is structured for modularity and **"Stateless"** operation:

| File | Purpose |
|------|---------|
| `src/engine.py` | Contains the `AetherisNet` class. Keeping the architecture separate allows the model to be imported into other projects (e.g., NIB OS integration). |
| `src/main.py` | The "Control Room." Manages the window lifecycle, UI rendering, and the image-to-tensor preprocessing pipeline. |
| `model/aetheris_model.pth` | The serialized neural weights, optimized for CPU inference. |

---

## 🚀 Installation & Deployment

### 1. Clone the Core

git clone https://github.com/NickIBrody/Aetheris-Neural-Engine
cd Aetheris-Neural-Engine



### 2. Prepare the Environment (Linux/Debian)

Aetheris requires specific system-level engines to render its terminal UI


sudo apt update && sudo apt install -y \
    python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.1


### 3. Install Python Dependencies

pip install requirements.txt



### 4. Ignite the Engine

python3 src/main.py


# 🎮 MNIST Digit Classifier — Neural Core Interface

> Hand-drawn digit recognition with a convolutional neural network.

## 🧠 Neural Core Init

Upon launch, the system verifies the model weights and initializes the graphical user interface (GUI).

## ✍️ Input Buffer

Use your mouse or touchpad to draw a digit (`0`–`9`) in the **black central terminal zone**.

## 🔍 Inference

Click the **`GUESS`** button. The image is:
- Captured from the canvas
- Resized to `28×28` pixels
- Normalized to MNIST standards
- Processed by the CNN

## 💬 Feedback

The AI responds in the chat interface with:
- Its predicted digit
- Confidence percentage (%)

---

## 📡 Future Roadmap

- [ ] **Real-time Prediction**  
  Removing the "Guess" button for live, frame-by-frame inference.

- [ ] **ONNX Export**  
  Converting the core to ONNX for an even lower memory footprint on edge devices.

- [ ] **Dataset Expansion**  
  Fine-tuning the engine to recognize custom mathematical symbols.

---

## ⚖️ License

MIT

---

---

> Built with **Python**, **PyTorch**, and **PyWebView**.




