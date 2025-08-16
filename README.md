# 🎯 VLC Gesture Controled  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)  [![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green)](https://developers.google.com/mediapipe)  [![OpenCV](https://img.shields.io/badge/OpenCV-4.x-yellow)](https://opencv.org/)  

**Gesture-controlled VLC using MediaPipe + Neural Network**  


## 📺 Demo
*(Coming soon)*  


---

## ✋ Features
| Gesture        | Action            | Trigger type         |
| -------------- | ----------------- | -------------------- |
| palm           | base/activation   | gating/activation    |
| fist           | play/pause toggle | one-shot per session |
| two_fingers    | mute/unmute       | one-shot per session |
| finger_up      | volume up         | continuous           |
| finger_down    | volume down       | continuous           |

> **Note:** `palm` arms the system for the next gesture.

---

## 🛠 How It Works (Architecture)
```mermaid
flowchart LR
    A[Camera] --> B[MediaPipe Hands<br>(landmark extraction)]
    B --> C[Preprocessing]
    C --> D[Neural Net Classifier<br>(PyTorch)]
    D --> E[Gesture FSM<br>(Palm-gated logic)]
    E --> F[VLC Controller<br>(Hotkeys / HTTP API)]
```
- **One-shot**: `fist`, `two_fingers` — triggered once per palm session.  
- **Continuous**: `finger_up`, `finger_down` — repeat until palm shown again or hand removed.

---

## 🎬 VLC Media Player Setup (Step-by-step)

### 1. Install VLC (Windows)
Download: [VLC Official Site](https://www.videolan.org/vlc/)  

### 2. Enable Control

#### **a) Hotkeys** *(default fallback)*  
- Space → Play/Pause  
- ↑ / ↓ → Volume Up/Down  
- M → Mute/Unmute  

#### **b) HTTP Interface** *(recommended for precision)*  
1. Open VLC → Tools → Preferences → Show All (bottom-left).  
2. Interface → Main Interfaces → Check **Web**.  
3. Expand Main Interfaces → **Lua** → Set *Lua HTTP* password (`vlcpassword` default).  
4. Default port: `8080`.  
5. Launch VLC with HTTP enabled:  
   ```bash
   vlc --intf http --http-password vlcpassword
   ```  
6. Test: Visit `http://localhost:8080` in browser.

---

## 💻 Project Installation & Setup
```bash
# Clone repo
git clone <your_repo_url>
cd <repo_name>

# (Recommended) Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:**  
```bash
pip install opencv-python mediapipe torch joblib psutil pyautogui requests
```

---

## 📦 Model: Training & Data Collection

### 📊 Dataset
- **Gestures**: 5 (`palm`, `fist`, `finger_up`, `finger_down`, `two_fingers`)  
- **Samples**: 200 live captures per gesture  
- **Augmentation** (×5 per sample):  
  - Rotation: ±15°  
  - Scaling: 0.9× – 1.1×  
  - Gaussian Noise: mean=0, std=0.01  
- **Total**: 1200 images per gesture → **6000 samples** overall  
- **Split**: 80% training, 20% validation  

### 🖐 Input Features
- 21 hand landmarks × 3 coordinates (x, y, z) = **63 features**  

### 🧠 Model Architecture (GestureNet)
- Input layer: 63 → 63  
- Hidden Layer 1: 63 → 128 (ReLU)  
- Hidden Layer 2: 128 → 128 (ReLU)  
- Output layer: 128 → 5  

### ⚙ Training Setup
- **Loss**: CrossEntropyLoss (multi-class classification)  
- **Optimizer**: Adam, lr = 0.001  
- **Epochs**: 50  
- **Batch size**: 32  
- **Device**: GPU (if available)  

### 📈 Performance (averages across epochs)
- Loss: **0.0275**  
- Accuracy: **99.10%**  
- Precision: **99.24%**  
- Recall: **99.11%**  
- F1 Score: **99.10%**  

Outputs:
- `models/gesture_model.pth`  
- `models/gesture_labels.pkl`  

---

## 👀 Test/Preview: Live Recognition
```bash
python code/recognize.py
```
Shows:
- Predicted gesture
- FPS
- Confidence score  

Press **ESC** to quit.  

---

## 🎯 VLC Control Runtime
```bash
python main.py
```
- Detects VLC media playback.  
- Prompts user to enable webcam gesture control.  
- Starts `gesture_vlc.py` for real-time control.  

Palm → Action mapping:
- Palm → Fist → Play/Pause  
- Palm → Two fingers → Mute/Unmute  
- Palm → Finger up → Volume up (continuous)  
- Palm → Finger down → Volume down (continuous)  

---

## 📂 File-by-File Explanation
- **train_model.py** — Data collection, augmentation, training.  
- **recognize.py** — Test trained model with webcam.  
- **gesture_vlc.py** — Main recognition + palm-gated gesture FSM.  
- **vlc_controller.py** — VLC control (HTTP API or hotkeys).  
- **main.py** — VLC monitor + gesture process manager.  
- **metrics_logger.py** — Logs training, inference, gesture usage.  
- **plot_training_metrics.py** — Plots curves (Loss, Accuracy, Precision, Recall, F1).  


## 🙏 Acknowledgements
- [MediaPipe Hands](https://developers.google.com/mediapipe)  
- [PyTorch](https://pytorch.org/)  
- [OpenCV](https://opencv.org/)  
- [VLC](https://www.videolan.org/vlc/)  
