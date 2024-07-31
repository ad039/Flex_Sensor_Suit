import cv2
import mediapipe as mp
import time
from bluepy import btle
import numpy as np
from threading import Thread, Event
import queue
import csv
import struct


# function to communicate with microcontroller and recieve sensor data over bluetooth
def ble_task(queue=queue.LifoQueue):
    global ready

    print("Connecting...")

    # MAC addresses for ble chips
    # "29:F0:E3:F9:C9:CD" for Arduino Nano BLE
    # "93:43:92:07:91:11" for Xioa nrf52 BLE 

    FlexSensorSuit = btle.Peripheral("29:F0:E3:F9:C9:CD")

    print("connected")
    FlexSensorSuit.getServices()
    flexSensorService = FlexSensorSuit.getServiceByUUID("0000fff0-0000-1000-8000-00805f9b34fb")
    print("connected to service")

    flexSensorCharValue = flexSensorService.getCharacteristics()[0]
    print("connected to characteristic")
    i = 0

    while True:
        # wait for writer to be ready
        writer_ready_evnt.wait()

        val = flexSensorCharValue.read()
        val = struct.unpack("<hhhhhhh",val)
        #print(val)
        queue.put(val)

        
        
# function to run pose detection using mediapipe
def pose_estimation(queue=queue.LifoQueue):
    global ready


    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    time_prev = 0
    i = 0
    


    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.1, model_complexity=2) as pose:
        
        while cap.isOpened():
            
            # wait for writer to be ready
            writer_ready_evnt.wait()

            success, image = cap.read()

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) # flip to get selfi view, this impacts the landmark extraction later on: take left hand for right hand

            image.flags.writeable = False
            
            results = pose.process(image)

            image.flags.writeable = True
            image_height, image_width, image_depth = image.shape

            hand_centre = (0, 0, 0)
            normal_vector = [0, 0, 0]

            if results.pose_landmarks is not None:
                right_shoulder = [results.pose_world_landmarks.landmark[11].x,results.pose_world_landmarks.landmark[11].y, results.pose_world_landmarks.landmark[11].z]
                right_wrist = [results.pose_world_landmarks.landmark[15].x,results.pose_world_landmarks.landmark[15].y, results.pose_world_landmarks.landmark[15].z]
                right_index = [results.pose_world_landmarks.landmark[19].x,results.pose_world_landmarks.landmark[19].y, results.pose_world_landmarks.landmark[19].z]
                #right_pinky = [results.pose_world_landmarks.landmark[17].x,results.pose_world_landmarks.landmark[17].y, results.pose_world_landmarks.landmark[17].z]
                
                hand_centre_world = [np.mean([right_wrist, right_index], axis=0)]

                #normal_vector = np.cross(np.subtract(hand_points[2],hand_points[0]), np.subtract(hand_points[1], hand_points[2]))
                #normal_vector /= np.linalg.norm(normal_vector)
                
                right_hand_norm = np.subtract(hand_centre_world, right_shoulder)

                #queue_array = np.append(right_hand_norm, normal_vector)
                #print(queue_array)
                queue.put(right_hand_norm)
                i+=1
                        
            #startpoint = (int(hand_centre[0]*image_width), int(hand_centre[1]*image_height))
            #endpoint = (int(50*normal_vector[0]+startpoint[0]), int(50*normal_vector[1]+startpoint[1]))
            #thickness = 9
            #color = (255, 0, 0)
            #image = cv2.line(image, startpoint, endpoint, color, thickness)
             
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Mediapipe pose', image)

            if cv2.waitKey(1) & 0xFF ==27:
                break

    cap.release()


# function to control reading the queues and sending that data to a csv file (output.csv)
def writer_task(ble_queue=queue.Queue, pose_queue=queue.Queue, writer=None):
    global prevTime

    # set writer ready event
    writer_ready_evnt.set()

    print(f'loop: {(time.perf_counter()-prevTime)*1000:.3f}')
    prevTime = time.perf_counter()
    # read the queues from BLE and Pose Estimation
    pose_val = pose_queue.get()
    ble_val = ble_queue.get()

    # close ready event
    writer_ready_evnt.clear()

    # empty the queues so they dont clog up
    with pose_queue.mutex:
        pose_queue.queue.clear()
    
    with ble_queue.mutex:
        ble_queue.queue.clear()

    # append all data into a  numpy array
    queue_val = np.append(time.time(), np.append(np.divide(ble_val, 100), pose_val))
    #print(queue_val)
    writer.writerow(queue_val)
    #print(f'duration: {(time.perf_counter()-prevTime)*1000:.3f}')


# function to control the speed of the writter loop, this in turn sets the maximum speed of the system
def do_every(period,f,*args):
    def g_tick():
        t = time.time()
        while True:
            t += period
            yield max(t - time.time(),0)
    g = g_tick()
    while True:
        time.sleep(next(g))
        f(*args)


if __name__ == "__main__":


    prevTime = 0.0

    # two Lifo queues to get the most recent data so data allignes better
    ble_q = queue.LifoQueue(maxsize=100)
    pose_q = queue.LifoQueue(maxsize=100)

    with open('./Pose_Estimation/output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp","ElbowFlex", "ShoulderFlex1", "ShoulderFlex2", "ShoulderFlex3", "ForearmFlex", "HandFlex1", "HandFlex2", "P_x", "P_y", "P_z", "O_x", "O_y", "O_z"])
        
        writer_ready_evnt = Event()

        ble_thread = Thread(target=ble_task, args=(ble_q,))
        pose_estimation_thread = Thread(target=pose_estimation, args=(pose_q,))
        writer_thread = Thread(target=do_every, args=(0.1, writer_task, ble_q, pose_q, writer))
        
        ble_thread.start()
        pose_estimation_thread.start()

        time.sleep(10)
        writer_thread.start()

        ble_thread.join()
        pose_estimation_thread.join()
        writer_thread.join()

