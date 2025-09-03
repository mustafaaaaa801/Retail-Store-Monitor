import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque
from ultralytics import YOLO

# إعداد MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# تصنيفات الحركة
actions = ["buy", "sell", "steal", "normal"]
data = []
labels = []

# تحميل نموذج YOLO للكشف عن الأشخاص
yolo_model = YOLO("yolo/yolov8n.pt")  # ضع yolov8n.pt في مجلد yolo/

for idx, action in enumerate(actions):
    folder = f"dataset/{action}"
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        cap = cv2.VideoCapture(path)
        person_sequences = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # كشف الأشخاص باستخدام YOLO
            results = yolo_model(frame, verbose=False)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size == 0:
                        continue

                    # تحويل الصورة إلى RGB ومعالجة MediaPipe Pose
                    rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(rgb)

                    if pose_results.pose_landmarks:
                        landmarks = []
                        for lm in pose_results.pose_landmarks.landmark:
                            landmarks.append([lm.x, lm.y])
                        flat_landmarks = np.array(landmarks).flatten()

                        # تحديد ID بسيط لكل شخص حسب موقع الصندوق
                        pid = f"{x1}_{y1}_{x2}_{y2}"
                        if pid not in person_sequences:
                            person_sequences[pid] = deque(maxlen=30)
                        person_sequences[pid].append(flat_landmarks)

        cap.release()

        # حفظ كل شخص كعينة مستقلة بعد تحويل deque إلى numpy array
        for pid, seq in person_sequences.items():
            if len(seq) >= 30:
                data.append(np.array(list(seq)))  # تصحيح الخطأ
                labels.append(idx)

# إنشاء مجلد extracted إذا لم يكن موجودًا
os.makedirs("extracted", exist_ok=True)
np.save("extracted/data.npy", np.array(data))
np.save("extracted/labels.npy", np.array(labels))
print("Data saved with multi-person support successfully!")
