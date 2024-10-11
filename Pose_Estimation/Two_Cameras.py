import cv2
import mediapipe as mp
import numpy as np
import time



mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0

with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9, model_complexity=1) as pose:

    while (cap1.isOpened() and cap2.isOpened()):
        
        success1, image1 = cap1.read()
        success2, image2 = cap2.read()


        image1 = cv2.rotate(image1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image2 = cv2.rotate(image2, cv2.ROTATE_90_COUNTERCLOCKWISE)

        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


        results1 = pose.process(image1)
        results2 = pose.process(image2)

        # Draw the pose annotations on the image.
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

        image1 = cv2.resize(image1, (340, 600))
        image2 = cv2.resize(image2, (340, 600))


        if results1.pose_landmarks:
            mp_drawing.draw_landmarks(image1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        if results2.pose_landmarks:
            mp_drawing.draw_landmarks(image2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # fps
        # font which we will be using to display FPS 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        # time when we finish processing for this frame 
        new_frame_time = time.time() 

        # Calculating the fps 

        # fps will be number of frame processed in given time frame 
        # since their will be most of time error of 0.001 second 
        # we will be subtracting it to get more accurate result 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 

        # converting the fps into integer 
        fps = int(fps) 

        # converting the fps to string so that we can display it on frame 
        # by using putText function 
        fps = str(fps) 

        # putting the FPS count on the frame 
        cv2.putText(image1, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        
        cv2.imshow('Mediapipe pose 1', image1)
        cv2.imshow('Mediapipe pose 2', image2)

        if cv2.waitKey(1) & 0xFF ==27:
            break

    cap1.release()
    cap2.release()
