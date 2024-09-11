import cv2
import mediapipe as mp
import numpy as np



mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9, model_complexity=1) as pose:

    while (cap1.isOpened() and cap2.isOpened()):
        
        success1, image1 = cap1.read()
        success2, image2 = cap2.read()

        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


        results1 = pose.process(image1)
        results2 = pose.process(image2)

        # Draw the pose annotations on the image.
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)


        if results1.pose_landmarks:
            mp_drawing.draw_landmarks(image1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        if results2.pose_landmarks:
            mp_drawing.draw_landmarks(image2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('Mediapipe pose 1', image1)
        cv2.imshow('Mediapipe pose 2', image2)

        if cv2.waitKey(1) & 0xFF ==27:
            break

    cap1.release()
    cap2.release()
