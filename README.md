# 🚀 FaceAuthSystem  
### Real-Time Face Recognition + Blink-Based Liveness + Attendance Logging

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![Mediapipe](https://img.shields.io/badge/Mediapipe-FaceMesh-orange)
![Torch](https://img.shields.io/badge/PyTorch-FaceNet-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

**FaceAuthSystem** is an AI-based real-time authentication system featuring:

- 🆔 **Face Enrollment (FaceNet Embeddings)**  
- 🎭 **Blink-Based Liveness Detection**  
- 👁️ **Face Recognition (Cosine Similarity)**  
- 📝 **Attendance Logging (IN/OUT)**  
- 🎥 **Live Webcam Feed using OpenCV**

Powered by **FaceNet, Mediapipe, OpenCV, PyTorch, NumPy, Pandas**.

---

## 🧠 Architecture
Webcam → Mediapipe Detection → Face Crop → FaceNet Embedding
→ Cosine Similarity → Identity Match → Blink Liveness
→ Attendance Logging (CSV)


---

## 📁 Project Structure

FaceAuthSystem/
│
├── src/
│ ├── enroll_user.py
│ ├── liveness.py
│ ├── main_system.py
│ └── utils/
│
├── embeddings/
├── logs/
├── models/
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE


---

## 🚀 Installation

### 1️⃣ Clone
```bash
git clone https://github.com/DhawalThombare37/FaceAuthSystem
cd FaceAuthSystem

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt

## 🧪 Usage
### Enroll User
```bash
python src/enroll_user.py

### Run Full System
```bash
python src/main_system.py

---
## Keyboard Controls
| Key | Action     |
| --- | ---------- |
| O   | Mark Entry |
| X   | Mark Exit  |
| Q   | Quit       |

---

##🧩 Tech Stack
~Face Detection → Mediapipe
~Liveness Detection → FaceMesh EAR
~Face Embedding → FaceNet (InceptionResnetV1)
~Matching → Cosine Similarity
~Logs → Pandas CSV

---

##📈 Future Enhancements
~Streamlit Web App
~Anti-Spoofing CNN
~Encrypted Embedding Storage
~Dashboard

---

##📝 License
MIT License.

---
##⭐ Support
If you found this useful, give the repo a ⭐ on GitHub!







