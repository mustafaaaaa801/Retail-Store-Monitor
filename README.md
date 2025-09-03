# Retail-Store-Monitor
AI-powered retail store monitoring system that detects multiple customers, tracks their body movements, and classifies actions as Buy, Sell, Steal, or Normal in real-time. Displays behavior labels in Arabic above each person using YOLOv8, Mediapipe, and LSTM.
 Retail Store Behavior Monitoring System

## Project Description
AI-powered retail store monitoring system that detects multiple customers, tracks their body movements, and classifies actions as Buy, Sell, Steal, or Normal in real-time. Behavior labels are displayed in Arabic above each person. Uses YOLOv8, Mediapipe Pose, and LSTM Neural Network.

---

## Project Structure

Retail-Store-Monitor/
│
├── models/
├── yolo/
├── extracted/
├── scripts/
├── dataset/
├── README.md
├── LICENSE
└── .gitignore

yaml
نسخ الكود

---

## Requirements

- Python 3.10+
- Libraries:

```bash
pip install opencv-python mediapipe tensorflow ultralytics numpy
How to Run
Data Collection: Place videos in dataset/{action} folders: buy, sell, steal, normal.
Extract pose data:

bash
نسخ الكود
python scripts/feature_extractor.py
Train the Model:

bash
نسخ الكود
python scripts/train_model.py
Run Live Monitoring:

bash
نسخ الكود
python scripts/multi_person_app.py
Press Esc to exit.

