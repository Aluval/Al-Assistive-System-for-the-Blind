# 🧠 AI Assistive System for the Blind

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9+-green.svg)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-YOLOv8-orange)
![AI](https://img.shields.io/badge/AI-Assistive%20Technology-purple)

An **AI-powered Assistive Navigation System** designed to help visually impaired individuals detect obstacles, understand surroundings, and receive real-time voice alerts using computer vision and sound analysis.

This project integrates **object detection, gesture recognition, facial expression detection, sound event detection, and direction guidance** into a single assistive system.

---

# 👨‍💻 Developer

**Developed by:**  
**Aluvala Ediga Harsha Vardhan Goud**

🎓 MCA Student  
🤖 AI / Machine Learning Enthusiast  
📍 India

---

# ⚠️ Important Notice

This project was **developed for educational and research purposes**.

If you use this code:

- You **must mention the original author**  
- Credit **Aluvala Ediga Harsha Vardhan Goud**

❗ Misleading others, removing the author name, or selling this code without permission may lead to **serious action**.

Please respect the developer's work and contribution.

---

# 🚀 Features

### 👁 Object Detection
Detects nearby objects using **YOLOv8** such as:

- Person
- Bicycle
- Car
- Bus
- Truck
- Animals
- Mobile phones

---

### 🚶 Approach Detection

Identifies if objects are **moving closer** using bounding box growth analysis.

Provides alerts like: 

Person approaching in front
Car on the left

---

### ✋ Gesture Recognition
```
Uses **MediaPipe Hands** to detect gestures:

- Open Palm
- Fist
- Thumbs Up
- Pointing
Example voice alert:
Person 1 gesture: Thumbs Up
```
---

### 🙂 Facial Expression Detection

Detects facial expressions using **Haarcascade classifiers**
```
Expressions detected:

- Smiling
- Neutral

Example output:
Person looks smiling
```
---
### 🚨 Emergency Sound Detection
```
Detects dangerous sound events:

- Sirens
- Alarms
- Loud impacts
Voice warnings:
Danger alarm detected
Loud noise detected. Be careful
```
---
### 🧭 Smart Direction Guidance
```
Divides camera view into:

- Left
- Center
- Right

Provides guidance like:
Left side is clear. Go left
Forward is clear. Go straight
All directions blocked
```

---

### 🔊 Voice Alerts

Uses **Text-to-Speech (pyttsx3)** to deliver real-time audio guidance.

This allows visually impaired users to navigate without needing a display.

---

# 🛠 Technologies Used

| Technology | Purpose |
|------------|---------|
Python | Core programming language |
YOLOv8 (Ultralytics) | Object detection |
OpenCV | Computer vision processing |
MediaPipe | Gesture recognition |
NumPy | Numerical computations |
PyTTSx3 | Voice alerts |
SoundDevice | Sound event detection |
Haarcascade | Facial expression detection |

---

# 🧠 AI / ML Concepts Used

- Object Detection
- Computer Vision
- Gesture Recognition
- Audio Signal Processing
- Real-Time AI Systems
- Assistive Technology

---

# ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/AI-Assistive-System-Blind.git
cd AI-Assistive-System-Blind
```
### 2️⃣ Install Dependencies
```pip install ultralytics opencv-python mediapipe numpy pyttsx3 sounddevice```
### 3️⃣ Run the Program
```python main.py```
Press Q to exit the system.
---
📂 Project Structure
```
AI-Assistive-System-Blind
│
├── main.py
├── yolov8n.pt
├── README.md
└── LICENSE
```
---
🎯 Applications
This system can be used for:
- Assistive technology for visually impaired people
- Smart navigation systems
- AI safety assistants
- Robotics navigation
- Smart wearable devices
---
📜 License
This project is licensed under the Apache License 2.0.

You may:

✔ Use
✔ Modify
✔ Distribute

However, proper attribution to the original developer is required.

Full license:
```https://www.apache.org/licenses/LICENSE-2.0```
---
⭐ Support
If you find this project useful:

⭐ Star the repository
🍴 Fork the project
📢 Share with developers and researchers
---
### ❤️ Acknowledgements
Special thanks to the open-source community:
- Ultralytics YOLO
- OpenCV
- MediaPipe
- Python AI ecosystem
---
© 2026 Aluvala Ediga Harsha Vardhan Goud
---





Provides alerts like:
