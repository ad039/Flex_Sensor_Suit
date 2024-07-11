import cv2
import mediapipe as mp
import time
from bluepy import btle
import numpy as np
import threading
from queue import Queue
import csv
import struct




def ble_startup(queue):
    global flexSensorCharValue
    print("Connecting...")

    # MAC addresses for ble chips
    # "29:F0:E3:F9:C9:CD" for Arduino Nano BLE
    # "93:43:92:07:91:11" for Xioa nrf52 BLE 

    FlexSensorSuit = btle.Peripheral("93:43:92:07:91:11")

    print("connected")
    FlexSensorSuit.getServices()
    flexSensorService = FlexSensorSuit.getServiceByUUID("0000fff0-0000-1000-8000-00805f9b34fb")
    print("connected to service")

    flexSensorCharValue = flexSensorService.getCharacteristics()[0]
    print("connected to characteristic")
    i = 0
    while True:
        val = flexSensorCharValue.read()
        val = struct.unpack("<hhhhhhh",val)
        #print(val)
        queue.put(val)
        i+= 1
        
        


def pose_estimation(queue):
    
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    time_prev = 0
    i = 0
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as holistic:
        
        while cap.isOpened():
            
            success, image = cap.read()

            start = time.time()

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) # flip to get selfi view, this impacts the landmark extraction later on: take left hand for right hand

            image.flags.writeable = False
            
            results = holistic.process(image)
            image_height, image_width, image_depth = image.shape

            hand_centre = (0, 0, 0)
            normal_vector = [0, 0, 0]

            if results.pose_landmarks is not None and results.left_hand_landmarks is not None:
                right_shoulder = [results.pose_world_landmarks.landmark[11].x,results.pose_world_landmarks.landmark[11].y, results.pose_world_landmarks.landmark[11].z]
                right_wrist = [results.pose_world_landmarks.landmark[15].x,results.pose_world_landmarks.landmark[15].y, results.pose_world_landmarks.landmark[15].z]
                right_index = [results.pose_world_landmarks.landmark[19].x,results.pose_world_landmarks.landmark[19].y, results.pose_world_landmarks.landmark[19].z]
                right_pinky = [results.pose_world_landmarks.landmark[17].x,results.pose_world_landmarks.landmark[17].y, results.pose_world_landmarks.landmark[17].z]
                hand_points = [[results.left_hand_landmarks.landmark[0].x, results.left_hand_landmarks.landmark[0].y, results.left_hand_landmarks.landmark[0].z],
                               [results.left_hand_landmarks.landmark[5].x, results.left_hand_landmarks.landmark[5].y, results.left_hand_landmarks.landmark[5].z],
                               [results.left_hand_landmarks.landmark[17].x, results.left_hand_landmarks.landmark[17].y, results.left_hand_landmarks.landmark[17].z]]
                
                hand_centre = np.mean(hand_points, axis=0)
                hand_centre_world = [np.mean([right_wrist, right_index], axis=0)]

                normal_vector = np.cross(np.subtract(hand_points[2],hand_points[0]), np.subtract(hand_points[1], hand_points[2]))
                normal_vector /= np.linalg.norm(normal_vector)
                
                right_hand_norm = np.subtract(hand_centre_world, right_shoulder)

                queue_array = np.append(right_hand_norm, normal_vector)
                #print(queue_array)
                queue.put(queue_array)
                i+=1
            
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            startpoint = (int(hand_centre[0]*image_width), int(hand_centre[1]*image_height))
            endpoint = (int(50*normal_vector[0]+startpoint[0]), int(50*normal_vector[1]+startpoint[1]))
            thickness = 9
            color = (255, 0, 0)
            image = cv2.line(image, startpoint, endpoint, color, thickness)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            #mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


            time_now = time.time()
            totalTime = time_now-time_prev
            time_prev = time_now
            fps = 1 / totalTime
            
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
    with open('Pose_Estimation/output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ElbowFlex", "ShoulderFlex1", "ShoulderFlex2", "ShoulderFlex3", "ForearmFlex", "HandFlex1", "HandFlex2", "P_x", "P_y", "P_z", "O_x", "O_y", "O_z"])
        while True:
            if time.perf_counter() - prevTime > 0.1:
                #print(f'loop: {(time.perf_counter()-prevTime)*1000:.3f}')
                prevTime = time.perf_counter()
                # read the queues from BLE and Pose Estimation
                ble_val = ble_queue.get()
                pose_val = pose_queue.get()
                queue_val = np.append(np.divide(ble_val,100), pose_val)
                #print(queue_val)
                writer.writerow(queue_val)
                #print(f'duration: {(time.perf_counter()-prevTime)*1000:.3f}')





if __name__ == "__main__":
    
    ble_q = Queue(maxsize=100)
    pose_q = Queue(maxsize=100)

    ble_thread = threading.Thread(target=ble_startup, args=(ble_q,))
    pose_estimation_thread = threading.Thread(target=pose_estimation, args=(pose_q,))
    writer_thread = threading.Thread(target=writer_task, args=(ble_q,pose_q))

    
    ble_thread.start()
    pose_estimation_thread.start()
    writer_thread.start()

    ble_thread.join()
    pose_estimation_thread.join()
    writer_thread.join()

