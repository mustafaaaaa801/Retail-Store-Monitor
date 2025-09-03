import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from collections import deque
from ultralytics import YOLO

# YOLO للكشف عن الأشخاص
yolo_model = YOLO("yolo/yolov8n.pt")

# تحميل نموذج السلوك
model = tf.keras.models.load_model("models/behavior_model.h5")
# تصنيفات بالعربية
actions = ["buy", "sell", "stealer", "normal"]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

sequences = {}  # person_id -> deque

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Store Monitor", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Store Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame, verbose=False)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb)

            if pose_results.pose_landmarks:
                landmarks = []
                for lm in pose_results.pose_landmarks.landmark:
                    landmarks.append([lm.x, lm.y])
                flat_landmarks = np.array(landmarks).flatten()

                # تحديد ID بسيط لكل شخص حسب موقع الصندوق
                person_id = (x1 + y1 + x2 + y2) % 1000
                if person_id not in sequences:
                    sequences[person_id] = deque(maxlen=30)
                sequences[person_id].append(flat_landmarks)

                # التنبؤ بالسلوك بعد 30 إطار
                if len(sequences[person_id]) == 30:
                    data_seq = np.expand_dims(sequences[person_id], axis=0)
                    pred = model.predict(data_seq, verbose=0)
                    label = np.argmax(pred)
                    action = actions[label]  # بالعربية

                    # إظهار النص فوق الشخص داخل الإطار بشكل دائم
                    text_x = x1
                    text_y = max(y1 - 20, 20)
                    cv2.putText(frame, f"{action}", (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # رسم المستطيل حول الشخص
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                mp_drawing.draw_landmarks(person_crop, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Store Monitor", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc للخروج
        break

cap.release()
cv2.destroyAllWindows()
