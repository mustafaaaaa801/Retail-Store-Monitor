import cv2
import os
import time

actions = ["buy", "sell", "steal", "normal"]
for action in actions:
    os.makedirs(f"dataset/{action}", exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Data Collector", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Data Collector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

recording = False
out = None
label = None

print("اضغط: 1=شراء  2=بيع  3=سرقة  4=عادي  q=خروج")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Data Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        label = "buy"
    elif key == ord('2'):
        label = "sell"
    elif key == ord('3'):
        label = "steal"
    elif key == ord('4'):
        label = "normal"
    elif key == ord('q'):
        break

    if label and not recording:
        filename = f"dataset/{label}/{int(time.time())}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        recording = True
        start_time = time.time()
        print(f"Recording {label}...")

    if recording:
        out.write(frame)
        if time.time() - start_time > 5:
            recording = False
            out.release()
            label = None
            print("Saved clip.")

cap.release()
cv2.destroyAllWindows()
