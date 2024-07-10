import cv2
import mediapipe as mp
import time
from bluepy import btle
import numpy as np
import threading
from queue import Queue
import csv




def ble_startup(queue):
    global flexSensorCharValue
    print("Connecting...")


    FlexSensorSuit = btle.Peripheral("29:F0:E3:F9:C9:CD")

    FlexSensorSuit.getServices()
    flexSensorService = FlexSensorSuit.getServiceByUUID("0000fff0-0000-1000-8000-00805f9b34fb")

    flexSensorCharValue = flexSensorService.getCharacteristics()[0]
    i = 0
    while True:
        val = flexSensorCharValue.read()
        queue.put(val)
        i += 1
        if 0xFF ==27:
            break


def pose_estimation(queue):
    
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    i = 100
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as holistic:
        
        while cap.isOpened():
            
            success, image = cap.read()

            start = time.time()

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) # flip to get selfi view, this impacts the landmark extraction later on: take left hand for right hand

            image.flags.writeable = False
            
            results = holistic.process(image)
            image_height, image_width, image_depth = image.shape

            if results.pose_landmarks is not None and results.left_hand_landmarks is not None:
                right_shoulder = [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].z]
                right_wrist = [results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST.value].y, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST.value].z]
                hand_points = [[results.left_hand_landmarks.landmark[0].x, results.left_hand_landmarks.landmark[0].y, results.left_hand_landmarks.landmark[0].z],
                               [results.left_hand_landmarks.landmark[5].x, results.left_hand_landmarks.landmark[5].y, results.left_hand_landmarks.landmark[5].z],
                               [results.left_hand_landmarks.landmark[17].x, results.left_hand_landmarks.landmark[17].y, results.left_hand_landmarks.landmark[17].z]]

                

                normal_vector = np.cross(np.subtract(hand_points[2],hand_points[0]), np.subtract(hand_points[1], hand_points[2]))
                normal_vector /= np.linalg.norm(normal_vector)

                print(normal_vector)
                right_wrist_norm = np.subtract(right_wrist,right_shoulder)

                
                
                queue.put(right_wrist_norm)
                i += 1
            
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


            end = time.time()
            totalTime = end-start
            fps = 1 / totalTime

            cv2.putText(image, f'{int(fps)}', (20,70), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0))

            cv2.imshow('Mediapipe pose', image)

            if cv2.waitKey(5) & 0xFF ==27:
                break

    cap.release()


def writer_task(ble_queue, pose_queue):
    prevTime = 0.0
    with open('output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        while True:
            if time.perf_counter() - prevTime > 0.1:
                #print(f'loop: {(time.perf_counter()-prevTime)*1000:.3f}')
                prevTime = time.perf_counter()
                # read the queues from BLE and Pose Estimation
                ble_val = ble_queue.get()
                pose_val = pose_queue.get()
                writer.writerow([ble_val, pose_val])
                #print(f'duration: {(time.perf_counter()-prevTime)*1000:.3f}')
            
            if 0xFF ==27:
                break





if __name__ == "__main__":
    
    ble_q = Queue(maxsize=100)
    pose_q = Queue(maxsize=100)

    #ble_thread = threading.Thread(target=ble_startup, args=(ble_q,))
    pose_estimation_thread = threading.Thread(target=pose_estimation, args=(pose_q,))
    writer_thread = threading.Thread(target=writer_task, args=(ble_q,pose_q))

    
    #ble_thread.start()
    pose_estimation_thread.start()
    writer_thread.start()

    #ble_thread.join()
    pose_estimation_thread.join()
    writer_thread.join()

