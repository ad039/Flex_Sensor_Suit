import cv2
import time
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

cap = cv2.VideoCapture(0)
prev_time = time.time()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    sucess, frame = cap.read()

    fps = 1/(time.time()-prev_time)
    prev_time = time.time()

    cv2.putText(frame, f'{int(fps)}', (20,70), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0))

    cv2.imshow('Window', frame)

    if cv2.waitKey(5) & 0xFF ==27:
        break

cap.release()
